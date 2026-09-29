from pathlib import Path
import time

import ollama
from PIL import Image


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_PATH = Path.home() / "Downloads" / "test.png"

MODEL = "minicpm-v4.6:1b"

NUMBER_OF_CHUNKS = 3

# Fraction of each chunk that overlaps with
# the neighbouring chunk.
OVERLAP = 0.20


PROMPT = """
Faithfully transcribe all handwritten scientific text
visible in this image section.

STRICT RULES:

- Transcribe only information actually visible.
- Do not invent information.
- Do not explain or interpret anything.
- Do not correct calculations.
- Preserve every number exactly.
- Preserve physical units exactly.
- Preserve mathematical symbols exactly.
- Convert mathematical expressions to LaTeX.
- Preserve Greek letters, subscripts, superscripts,
  exponents and fractions.
- Pay particular attention to exponents and powers of ten.
- If a character or number is uncertain, write [unclear].
- Do not reconstruct content outside this image section.

Return only the transcription.
"""


# --------------------------------------------------
# CREATE OVERLAPPING CHUNKS
# --------------------------------------------------

def create_chunks(
    image: Image.Image,
    number_of_chunks: int,
    overlap: float,
):
    width, height = image.size

    # Base height if the page were split normally.
    base_height = height / number_of_chunks

    # Add overlap on both sides of interior chunks.
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

        chunk = image.crop(
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
                "image": chunk,
                "top": top,
                "bottom": bottom,
            }
        )

    return chunks


# --------------------------------------------------
# TRANSCRIBE ONE CHUNK
# --------------------------------------------------

def transcribe_chunk(
    image_path: Path,
    chunk_number: int,
):

    print()
    print("=" * 60)
    print(f"TRANSCRIBING CHUNK {chunk_number}")
    print("=" * 60)

    start = time.time()

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": PROMPT,
                "images": [
                    str(image_path)
                ],
            }
        ],
        options={
            "temperature": 0,
        },
        keep_alive="5m",
    )

    elapsed = time.time() - start

    text = response[
        "message"
    ][
        "content"
    ]

    print(
        f"Time: {elapsed:.1f} seconds"
    )

    print()
    print(text)

    return text


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    print(
        f"Original image: {image.size[0]} x "
        f"{image.size[1]}"
    )

    chunks = create_chunks(
        image=image,
        number_of_chunks=NUMBER_OF_CHUNKS,
        overlap=OVERLAP,
    )

    output_directory = Path(
        "vision_chunks"
    )

    output_directory.mkdir(
        exist_ok=True
    )

    results = []

    total_start = time.time()

    for chunk in chunks:

        chunk_number = chunk["index"]

        chunk_path = (
            output_directory
            / f"chunk_{chunk_number}.jpg"
        )

        chunk["image"].save(
            chunk_path,
            format="JPEG",
            quality=95,
        )

        print()
        print(
            f"Chunk {chunk_number}: "
            f"y={chunk['top']} → "
            f"{chunk['bottom']}"
        )

        print(
            "Resolution:",
            chunk["image"].size,
        )

        transcription = transcribe_chunk(
            image_path=chunk_path,
            chunk_number=chunk_number,
        )

        results.append(
            {
                "chunk": chunk_number,
                "text": transcription,
            }
        )

    total_time = (
        time.time()
        - total_start
    )

    print()
    print("=" * 60)
    print("ALL CHUNKS")
    print("=" * 60)

    for result in results:

        print()
        print(
            f"--- CHUNK {result['chunk']} ---"
        )

        print(
            result["text"]
        )

    print()
    print(
        f"TOTAL TIME: {total_time:.1f} seconds"
    )


if __name__ == "__main__":
    main()
