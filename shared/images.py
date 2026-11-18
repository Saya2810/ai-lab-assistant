"""Image helpers shared by the assistant (handwritten notes) and the chatbot.

Phone photos are often 4000 px and more. Vision models work on a few hundred
to ~1500 px internally, so sending the full resolution only costs memory and
upload time. Every image is therefore scaled down before it is encoded.
"""

from __future__ import annotations

import io
import logging

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif")
MAX_SIDE = 1600


class ImageError(ValueError):
    """The bytes are not an image we can decode."""


def is_image_name(name: str) -> bool:
    return str(name).lower().endswith(IMAGE_SUFFIXES)


def _register_heif() -> None:
    try:  # optional: only needed for iPhone HEIC photos
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        log.info("pillow-heif not installed; HEIC/HEIF images are not supported")


def load_image(content: bytes):
    """Decode image bytes; applies the EXIF orientation (phone photos are often rotated)."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    _register_heif()
    try:
        image = Image.open(io.BytesIO(content))
        image = ImageOps.exif_transpose(image)
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError(
            "This image format cannot be read. For HEIC/HEIF photos install pillow-heif "
            "(pip install pillow-heif), or export the photo as JPEG/PNG."
        ) from exc
    return image.convert("RGB") if image.mode != "RGB" else image


def downscale(image, max_side: int = MAX_SIDE):
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    factor = max_side / longest
    from PIL import Image

    return image.resize((max(1, int(width * factor)), max(1, int(height * factor))), Image.LANCZOS)


def horizontal_strips(image, count: int, overlap: float = 0.2) -> list[dict]:
    """Full-width strips that overlap, so no line of text is cut in half."""
    width, height = image.size
    base = height / count
    pad = int(base * overlap)
    strips = []
    for index in range(count):
        top = max(0, int(index * base) - pad)
        bottom = min(height, int((index + 1) * base) + pad)
        strips.append(
            {"index": index + 1, "image": image.crop((0, top, width, bottom)), "top": top, "bottom": bottom}
        )
    return strips


def jpeg_bytes(image, quality: int = 90, max_side: int = MAX_SIDE) -> bytes:
    buffer = io.BytesIO()
    downscale(image, max_side).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
