import time
import ollama

IMAGE = "/Users/saya/Downloads/test.png"

start = time.time()

response = ollama.chat(
    model="qwen3-vl:8b",
    messages=[
        {
            "role": "user",
            "content": (
                "Transcribe the scientific laboratory "
                "notes in this image. Do not add or "
                "interpret information."
            ),
            "images": [IMAGE],
        }
    ],
    options={
        "temperature": 0,
    },
    keep_alive=0,
)

print(
    "\nTIME:",
    round(time.time() - start, 1),
    "seconds",
)

print(
    "\nRESULT:\n",
    response["message"]["content"],
)

print("\nFULL RESPONSE:")
print(response)

print("\nCONTENT:")
print(repr(response["message"]["content"]))
