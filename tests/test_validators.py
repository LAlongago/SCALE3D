from pathlib import Path

from shared.validators import validate_image_paths, validate_pointcloud_path


def test_validate_pointcloud_path_ascii(tmp_path: Path):
    path = tmp_path / "sample.ply"
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
                "1 1 1",
            ]
        ),
        encoding="utf-8",
    )
    result = validate_pointcloud_path(path)
    assert result.vertex_count == 2
    assert result.format_name == "ascii"


def test_validate_image_paths_reject_mismatched_sizes(tmp_path: Path):
    from PIL import Image

    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.new("RGB", (10, 10), "red").save(a)
    Image.new("RGB", (12, 10), "blue").save(b)
    try:
        validate_image_paths([a, b])
    except ValueError as exc:
        assert "same width and height" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected mismatched image sizes to fail.")
