from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pointcept inference in an external Python environment.")
    parser.add_argument("--ut-root", type=Path, required=True)
    parser.add_argument("--pointcept-root", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--weight-file", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--part-names-json", type=Path, required=True)
    parser.add_argument("--palette-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser.parse_args()


def ensure_on_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def save_prediction_ply(coord: np.ndarray, pred: np.ndarray, palette: np.ndarray, output_path: Path) -> None:
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


def main() -> int:
    args = parse_args()
    ensure_on_path(args.pointcept_root)

    import torch
    import torch.nn.functional as F

    from pointcept.datasets import build_dataset, collate_fn
    from pointcept.engines.defaults import default_config_parser, default_setup
    from pointcept.models import build_model

    palette = np.asarray(json.loads(args.palette_json.read_text(encoding="utf-8")), dtype=np.uint8)
    part_names = json.loads(args.part_names_json.read_text(encoding="utf-8"))

    cfg = default_config_parser(str(args.config_file.resolve()), None)
    cfg.weight = str(args.weight_file.resolve())
    cfg.save_path = str((args.output_dir / "pointcept_run").resolve())
    cfg.num_worker = 0
    cfg.batch_size_test = 1
    cfg.data.test.data_root = str(args.dataset_root.resolve())
    cfg.data.test.split = "infer"
    cfg.data.num_classes = args.num_classes
    cfg = default_setup(cfg)

    model = build_model(cfg.model)
    checkpoint = torch.load(cfg.weight, map_location="cpu", weights_only=False)
    weight = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        normalized = key[7:] if key.startswith("module.") else key
        weight[normalized] = value
    model.load_state_dict(weight, strict=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Pointcept inference currently requires a CUDA-capable environment.")
    model = model.cuda()
    model.eval()

    dataset = build_dataset(cfg.data.test)
    target_index = None
    for idx in range(len(dataset.data_list)):
        if os.path.basename(dataset.data_list[idx]) == args.sample_name:
            target_index = idx
            break
    if target_index is None:
        raise FileNotFoundError(
            f"Prepared sample '{args.sample_name}' was not found in dataset root {args.dataset_root}"
        )

    data_dict = dataset[target_index]
    fragment_list = data_dict["fragment_list"]
    segment = data_dict["segment"]
    pred_scores = torch.zeros((segment.size, cfg.data.num_classes), device="cuda")
    for fragment in fragment_list:
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

    if "inverse" in data_dict:
        pred_scores = pred_scores[data_dict["inverse"]]
    pred = pred_scores.argmax(dim=1).cpu().numpy().astype(np.int32)
    confidence = pred_scores.max(dim=1).values.cpu().numpy().astype(np.float32)

    sample_dir = args.dataset_root / "infer" / args.sample_name
    coord = np.load(sample_dir / "coord.npy")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.output_dir / f"{args.sample_name}_pred.npy"
    confidence_path = args.output_dir / f"{args.sample_name}_confidence.npy"
    summary_path = args.output_dir / f"{args.sample_name}_segmentation_summary.json"
    ply_path = args.output_dir / f"{args.sample_name}_pred.ply"
    np.save(pred_path, pred)
    np.save(confidence_path, confidence)
    save_prediction_ply(coord, pred, palette, ply_path)

    per_part_summary = []
    for part_id in range(args.num_classes):
        mask = pred == part_id
        count = int(mask.sum())
        per_part_summary.append(
            {
                "part_id": part_id,
                "part_name": part_names[str(part_id)],
                "point_count": count,
                "confidence": float(confidence[mask].mean()) if count > 0 else None,
                "status": "detected" if count > 0 else "missing",
            }
        )
    summary_path.write_text(json.dumps(per_part_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(
        json.dumps(
            {
                "pred_npy": str(pred_path.resolve()),
                "confidence_npy": str(confidence_path.resolve()),
                "summary_json": str(summary_path.resolve()),
                "segmentation_ply": str(ply_path.resolve()),
                "coord_npy": str((sample_dir / "coord.npy").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
