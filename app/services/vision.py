import io
import os
import tempfile
import time
from pathlib import Path

import ollama
from PIL import Image
from pillow_heif import register_heif_opener


register_heif_opener()


VISION_MODEL = "minicpm-v4.6:1b"

NUMBER_OF_CHUNKS = 3
OVERLAP = 0.20


VISION_PROMPT = """
Faithfully transcribe all handwritten scientific text
visible in this image section.

STRICT RULES:

1. Transcribe only information actually visible.
2. Do not invent information.
3. Do not explain or interpret anything.
4. Do not correct calculations.
5. Preserve every number exactly.
6. Preserve physical units exactly.
7. Preserve mathematical symbols exactly.
8. Convert mathematical expressions to LaTeX.
9. Preserve Greek letters, subscripts, superscripts,
   exponents and fractions using LaTeX.
10. Pay particular attention to exponents and powers of ten.
11. If a character, number or symbol is uncertain,
    write [unclear].
12. Do not reconstruct content outside this image section.

Return only the transcription.
"""


def load_image(
    image_content: bytes,
) -> Image.Image:
    """
    Load an uploaded image from bytes.

    HEIC and HEIF are supported through pillow-heif.
    """

    image = Image.open(
        io.BytesIO(image_content)
    )

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def create_chunks(
    image: Image.Image,
    number_of_chunks: int = NUMBER_OF_CHUNKS,
    overlap: float = OVERLAP,
) -> list[dict]:
    """
    Split an image vertically into overlapping
    horizontal sections.

    The full image width is preserved.
    """

    width, height = image.size

    base_height = (
        height / number_of_chunks
    )

    overlap_pixels = int(
        base_height * overlap
    )

    chunks = []

    for index in range(number_of_chunks):

        normal_top = int(
            index * base_height
        )

        normal_bottom = int(
            (index + 1) * base_height
        )

        top = max(
            0,
            normal_top - overlap_pixels,
        )

        bottom = min(
            height,
            normal_bottom + overlap_pixels,
        )

        chunk_image = image.crop(
            (
                0,
                top,
                width,
                bottom,
            )
        )

        chunks.append(
            {
                "index": index + 1,
                "image": chunk_image,
                "top": top,
                "bottom": bottom,
            }
        )

    return chunks


def save_temporary_chunk(
    image: Image.Image,
) -> str:
    """
    Save a chunk as a temporary JPEG.

    Returns the temporary file path.
    """

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg",
    )

    image.save(
        temp_file,
        format="JPEG",
        quality=95,
    )

    temp_file.close()

    return temp_file.name


def transcribe_chunk(
    chunk: dict,
    is_last_chunk: bool,
) -> str:
    """
    Transcribe one image chunk with MiniCPM.
    """

    chunk_number = chunk["index"]

    image_path = save_temporary_chunk(
        chunk["image"]
    )

    try:

        print()
        print(
            f"Vision chunk {chunk_number}: "
            f"y={chunk['top']} -> "
            f"{chunk['bottom']}"
        )

        print(
            "Chunk resolution:",
            chunk["image"].size,
        )

        start = time.time()

        response = ollama.chat(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": VISION_PROMPT,
                    "images": [
                        image_path
                    ],
                }
            ],
            options={
                "temperature": 0,
            },

            # Keep MiniCPM loaded between chunks.
            # Unload it after the final chunk.
            keep_alive=(
                0
                if is_last_chunk
                else "5m"
            ),
        )

        elapsed = (
            time.time() - start
        )

        print(
            f"Chunk {chunk_number} inference: "
            f"{elapsed:.1f} seconds"
        )

        return (
            response["message"]["content"]
            .strip()
        )

    finally:

        if os.path.exists(
            image_path
        ):
            os.remove(
                image_path
            )


def combine_transcriptions(
    transcriptions: list[dict],
) -> str:
    """
    Combine chunk transcriptions without allowing
    another AI model to alter scientific values.

    Overlapping text is deliberately preserved for
    now so that conflicting OCR results remain visible
    to the user.
    """

    sections = []

    for result in transcriptions:

        sections.append(
            (
                f"--- IMAGE SECTION "
                f"{result['index']} ---\n"
                f"{result['text']}"
            )
        )

    return "\n\n".join(
        sections
    )


def analyze_lab_image(
    image_content: bytes,
) -> str:
    """
    Transcribe a laboratory-note image using
    overlapping image chunks and MiniCPM.

    The returned text is intended to be reviewed
    and edited by the user before it is passed to
    the laboratory-entry generation model.
    """

    total_start = time.time()

    image = load_image(
        image_content
    )

    print()
    print(
        "Original image:",
        image.size,
    )

    chunks = create_chunks(
        image=image,
    )

    transcriptions = []

    for position, chunk in enumerate(
        chunks
    ):

        is_last_chunk = (
            position
            == len(chunks) - 1
        )

        text = transcribe_chunk(
            chunk=chunk,
            is_last_chunk=is_last_chunk,
        )

        transcriptions.append(
            {
                "index": chunk["index"],
                "text": text,
            }
        )

    combined_text = (
        combine_transcriptions(
            transcriptions
        )
    )

    total_time = (
        time.time()
        - total_start
    )

    print()
    print(
        "Total vision processing:",
        f"{total_time:.1f} seconds",
    )

    return combined_text