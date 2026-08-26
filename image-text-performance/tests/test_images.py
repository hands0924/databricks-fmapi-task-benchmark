"""이미지 → data URL 변환 테스트 (src/adapters/images.py)."""

import base64
import io

import pytest
from PIL import Image
from src.adapters.images import bytes_to_data_url, pil_to_data_url


def decode(data_url):
    header, b64 = data_url.split(",", 1)
    return header, Image.open(io.BytesIO(base64.b64decode(b64)))


def test_returns_jpeg_data_url():
    header, img = decode(pil_to_data_url(Image.new("RGB", (10, 10), "red")))
    assert header == "data:image/jpeg;base64"
    assert img.format == "JPEG"
    assert img.size == (10, 10)


def test_downscales_long_side_keeping_aspect_ratio():
    _, img = decode(pil_to_data_url(Image.new("RGB", (2000, 1000)), max_side=768))
    assert img.size == (768, 384)


def test_small_image_is_not_upscaled():
    _, img = decode(pil_to_data_url(Image.new("RGB", (100, 50)), max_side=768))
    assert img.size == (100, 50)


def test_extreme_aspect_ratio_keeps_at_least_one_pixel():
    _, img = decode(pil_to_data_url(Image.new("RGB", (5000, 3)), max_side=100))
    assert img.size == (100, 1)


def test_rgba_is_converted_for_jpeg_compatibility():
    _, img = decode(pil_to_data_url(Image.new("RGBA", (8, 8), (0, 255, 0, 128))))
    assert img.mode == "RGB"


def test_palette_mode_is_converted():
    _, img = decode(pil_to_data_url(Image.new("P", (8, 8))))
    assert img.mode == "RGB"


def test_grayscale_mode_is_preserved():
    _, img = decode(pil_to_data_url(Image.new("L", (8, 8))))
    assert img.mode == "L"


def test_png_format_sets_matching_mime():
    header, img = decode(pil_to_data_url(Image.new("RGB", (8, 8)), fmt="PNG"))
    assert header == "data:image/png;base64"
    assert img.format == "PNG"


@pytest.mark.parametrize("bad", ["not-an-image", b"bytes", 42, None])
def test_non_pil_input_raises_type_error(bad):
    with pytest.raises(TypeError, match="PIL.Image"):
        pil_to_data_url(bad)


def test_bytes_to_data_url_defaults_to_jpeg():
    expected = base64.b64encode(b"hello").decode()
    assert bytes_to_data_url(b"hello") == f"data:image/jpeg;base64,{expected}"


def test_bytes_to_data_url_honors_mime():
    assert bytes_to_data_url(b"x", mime="image/png").startswith("data:image/png;base64,")
