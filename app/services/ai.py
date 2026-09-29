import json

import ollama


MODEL = "qwen2.5:7b"


def generate_text(prompt: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI laboratory assistant. "
                    "Convert laboratory observations into "
                    "clear, precise scientific documentation. "
                    "Never invent measurements, procedures, "
                    "observations, explanations, conclusions, "
                    "or results that were not provided by the user."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response["message"]["content"]


def generate_lab_entry(
    raw_notes: str,
    templates: list | None = None,
    categories: list | None = None,
    statuses: list | None = None,
    resources: list | None = None,
    tags: list | None = None,
) -> dict:
    """
    Convert reviewed source material into a structured
    laboratory entry.

    eLabFTW organizational fields are deliberately NOT
    selected by the AI. The user chooses these manually
    in the form.
    """

    system_prompt = r"""
You are a scientific laboratory notebook assistant.

Convert the supplied source material into a concise,
well-structured laboratory notebook entry.

The source may contain written notes, audio transcripts,
image transcriptions, or several combined sources.

SCIENTIFIC INTEGRITY

- Use only information explicitly present in the source.
- Never invent, infer, correct, calculate, or complete information.
- Preserve all numbers, signs, exponents, units, equations,
  observations, and uncertainties exactly as provided.
- Never replace a value with a known or more plausible value.
- Never correct a calculation, even if it appears scientifically wrong.
- Preserve [unclear] wherever it occurs.
- If sources conflict, preserve the conflicting information.
- Do not choose which version is correct.
- Remove only clear duplication caused by repeated source content.
- Filenames and source labels are metadata, not experimental facts.

ELABFTW STRUCTURE

The AI must NOT select organizational fields.
The user selects these manually.

Always return:

"template": null
"category": null
"status": null
"resource": null
"tags": []

Do not infer these fields from the scientific content.

LAB NOTE FORMATTING

Write "lab_notes" as clean Markdown suitable for eLabFTW.

Use:
- ## for major sections
- ### for subsections
- normal Markdown lists where useful
- concise scientific language

For mathematics, use LaTeX inside dollar delimiters ONLY.

Inline mathematics:

$E = mc^2$

Display mathematics:

$$
v = \sqrt{\frac{2GM}{R}}
$$

IMPORTANT:

- NEVER use \( ... \)
- NEVER use \[ ... \]
- NEVER wrap the Markdown in ```markdown or any other code fence.
- NEVER put equations inside Markdown code blocks.
- NEVER output literal Markdown code fences.
- Every LaTeX expression must be inside $...$ or $$...$$.
- Preserve the mathematical content exactly as supplied.
- Formatting may change; scientific content may not.

MEASUREMENTS

Only add a value to "measurements" when the source explicitly
identifies it as an experimental measurement.

Do not treat equations, example calculations, theoretical constants,
or calculated values as measurements.

Measurement format:

{
  "Wavelength": {
    "value": 532,
    "unit": "nm"
  }
}

OUTPUT

Return valid JSON only.

Use exactly this structure:

{
  "title": "short factual title",
  "lab_notes": "Markdown laboratory notes",
  "template": null,
  "category": null,
  "status": null,
  "resource": null,
  "tags": [],
  "measurements": {}
}

Do not include any text before or after the JSON object.
"""

    user_prompt = f"""
SOURCE MATERIAL:

--- BEGIN SOURCE ---

{raw_notes}

--- END SOURCE ---

Create the laboratory notebook entry.

Remember:
- preserve scientific content exactly
- do not repair OCR errors
- do not infer missing information
- use $...$ and $$...$$ for all mathematics
- do not use code fences
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        format="json",
        options={
            "temperature": 0,
        },
    )

    content = response["message"]["content"]

    try:
        result = json.loads(content)

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "The AI did not return valid JSON."
        ) from error

    # --------------------------------------------------
    # SECURITY / RELIABILITY GUARD
    #
    # Even if the model ignores the prompt, it cannot
    # automatically classify the experiment.
    # --------------------------------------------------

    result["template"] = None
    result["category"] = None
    result["status"] = None
    result["resource"] = None
    result["tags"] = []

    # Make sure expected fields exist.
    result.setdefault(
        "title",
        "Laboratory entry",
    )

    result.setdefault(
        "lab_notes",
        "",
    )

    result.setdefault(
        "measurements",
        {},
    )

    return result


def generate_lab_entry_from_elab(
    raw_notes: str,
    templates: list,
    categories: list,
    statuses: list,
    resources: list,
    tags: list,
) -> dict:
    """
    Compatibility wrapper used by the current UI.

    The eLabFTW structure is intentionally not supplied
    to the language model anymore. The user chooses the
    organizational structure manually.
    """

    return generate_lab_entry(
        raw_notes=raw_notes,
    )