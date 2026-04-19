from __future__ import annotations

import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

from shared.palette import PALETTE_36
from shared.product_models import get_product_model

logger = logging.getLogger("scale3d.pointcept")


def _ensure_pointcept_on_path(ut_root: Path) -> Path:
    pointcept_root = ut_root / "Pointcept"
    if str(pointcept_root) not in sys.path:
        sys.path.insert(0, str(pointcept_root))
    return pointcept_root


def _save_prediction_ply(coord: np.ndarray, pred: np.ndarray, output_path: Path) -> None:
    palette = np.asarray(PALETTE_36, dtype=np.uint8)
    colors = palette[np.clip(pred.astype(np.int64), 0, len(palette) - 1)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {coord.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("property int pred_label\n")
        handle.write("end_header\n")
        for xyz, rgb, label in zip(coord, colors, pred):
            handle.write(
                f"{float(xyz[0]):.6f} {float(xyz[1]):.6f} {float(xyz[2]):.6f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} {int(label)}\n"
            )


def run_pointcept_inference(
    ut_root: Path,
    product_model_id: str,
    dataset_root: Path,
    sample_name: str,
    output_dir: Path,
) -> dict[str, Path]:
    product_model = get_product_model(product_model_id)
    pointcept_root = _ensure_pointcept_on_path(ut_root)
    logger.info(
        "Preparing Pointcept inference: product_model_id=%s dataset_root=%s sample_name=%s output_dir=%s",
        product_model_id,
        dataset_root,
        sample_name,
        output_dir,
    )

    import torch
    import torch.nn.functional as F

    from pointcept.datasets import build_dataset, collate_fn
    from pointcept.engines.defaults import default_config_parser, default_setup
    from pointcept.models import build_model

    cfg = default_config_parser(
        str((ut_root / product_model.pointcept_model_config).resolve()),
        None,
    )
    logger.info("Loaded Pointcept config template: %s", ut_root / product_model.pointcept_model_config)
    cfg.weight = str((ut_root / product_model.pointcept_weight_path).resolve())
    cfg.save_path = str((output_dir / "pointcept_run").resolve())
    cfg.num_worker = 0
    cfg.batch_size_test = 1
    cfg.data.test.data_root = str(dataset_root.resolve())
    cfg.data.test.split = "infer"
    cfg.data.num_classes = product_model.num_parts
    cfg = default_setup(cfg)
    logger.info("Pointcept runtime config prepared. weight=%s save_path=%s", cfg.weight, cfg.save_path)

    logger.info("Building Pointcept model.")
    model = build_model(cfg.model)
    logger.info("Loading checkpoint from %s", cfg.weight)
    checkpoint = torch.load(cfg.weight, map_location="cpu", weights_only=False)
    weight = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        normalized = key[7:] if key.startswith("module.") else key
        weight[normalized] = value
    model.load_state_dict(weight, strict=True)
    logger.info("Checkpoint loaded successfully. parameter_groups=%s", len(weight))
    if not torch.cuda.is_available():
        raise RuntimeError("Pointcept inference currently requires a CUDA-capable environment.")
    logger.info("CUDA detected. device_count=%s current_device=%s", torch.cuda.device_count(), torch.cuda.current_device())
    logger.info("Moving Pointcept model to CUDA.")
    model = model.cuda()
    model.eval()
    logger.info("Pointcept model is now in eval mode on CUDA.")

    logger.info("Building Pointcept dataset for inference.")
    dataset = build_dataset(cfg.data.test)
    logger.info("Dataset built successfully. total_samples=%s", len(dataset.data_list))
    target_index = None
    for idx in range(len(dataset.data_list)):
        if os.path.basename(dataset.data_list[idx]) == sample_name:
            target_index = idx
            break
    if target_index is None:
        raise FileNotFoundError(f"Prepared sample '{sample_name}' was not found in dataset root {dataset_root}")
    logger.info("Resolved sample index: sample_name=%s target_index=%s", sample_name, target_index)

    logger.info("Loading sample data_dict from dataset.")
    data_dict = dataset[target_index]
    fragment_list = data_dict["fragment_list"]
    segment = data_dict["segment"]
    logger.info(
        "Sample loaded. fragment_count=%s point_count=%s has_inverse=%s",
        len(fragment_list),
        segment.size,
        "inverse" in data_dict,
    )
    pred_scores = torch.zeros((segment.size, cfg.data.num_classes), device="cuda")
    for fragment_index, fragment in enumerate(fragment_list, start=1):
        logger.info(
            "Running Pointcept fragment inference %s/%s for sample %s.",
            fragment_index,
            len(fragment_list),
            sample_name,
        )
        input_dict = collate_fn([fragment])
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.cuda(non_blocking=True)
        idx_part = input_dict["index"]
        with torch.no_grad():
            pred_part = model(input_dict)["seg_logits"]
            pred_part = F.softmax(pred_part, dim=-1)
            start = 0
            for end in input_dict["offset"]:
                pred_scores[idx_part[start:end], :] += pred_part[start:end]
                start = end
        logger.info(
            "Finished fragment inference %s/%s for sample %s. fragment_points=%s",
            fragment_index,
            len(fragment_list),
            sample_name,
            int(idx_part.shape[0]),
        )

    if "inverse" in data_dict:
        logger.info("Applying inverse index remapping for sample %s.", sample_name)
        pred_scores = pred_scores[data_dict["inverse"]]
    pred = pred_scores.argmax(dim=1).cpu().numpy().astype(np.int32)
    confidence = pred_scores.max(dim=1).values.cpu().numpy().astype(np.float32)
    logger.info("Prediction tensors converted to numpy. points=%s", pred.shape[0])

    sample_dir = dataset_root / "infer" / sample_name
    coord = np.load(sample_dir / "coord.npy")
    logger.info("Loaded coord.npy from %s", sample_dir / "coord.npy")

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{sample_name}_pred.npy"
    confidence_path = output_dir / f"{sample_name}_confidence.npy"
    summary_path = output_dir / f"{sample_name}_segmentation_summary.json"
    ply_path = output_dir / f"{sample_name}_pred.ply"
    np.save(pred_path, pred)
    np.save(confidence_path, confidence)
    _save_prediction_ply(coord, pred, ply_path)
    logger.info(
        "Saved Pointcept outputs: pred=%s confidence=%s ply=%s summary=%s",
        pred_path,
        confidence_path,
        ply_path,
        summary_path,
    )

    per_part_summary = []
    for part_id in range(product_model.num_parts):
        mask = pred == part_id
        count = int(mask.sum())
        per_part_summary.append(
            {
                "part_id": part_id,
                "part_name": product_model.part_names[str(part_id)],
                "point_count": count,
                "confidence": float(confidence[mask].mean()) if count > 0 else None,
                "status": "detected" if count > 0 else "missing",
            }
        )
    summary_path.write_text(json.dumps(per_part_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Pointcept inference completed successfully for sample %s.", sample_name)
    return {
        "pred_npy": pred_path,
        "confidence_npy": confidence_path,
        "summary_json": summary_path,
        "segmentation_ply": ply_path,
        "coord_npy": sample_dir / "coord.npy",
    }
