import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

import config

load_dotenv()


def get_openai_client() -> OpenAI:
    """
    Create Groq client (OpenAI-compatible API).

    Returns:
        OpenAI client pointed at Groq.
    """
    return OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=config.LLM_BASE_URL
    )


def get_structured_completion(
    prompt: str,
    response_model: type[BaseModel],
    model: str | None = None,
    retries: int = 2
) -> BaseModel:
    """
    Generate structured output.

    Args:
        prompt: Input prompt.
        response_model: Pydantic response model.
        model: Groq model name.
        retries: Attempts before giving up on invalid JSON.

    Returns:
        Parsed response model.
    """
    model = model or config.LLM_MODEL

    client = get_openai_client()

    schema = json.dumps(response_model.model_json_schema(), indent=2)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert financial analyst. "
                "Respond with a single JSON object matching this JSON schema:\n"
                f"{schema}"
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    for attempt in range(retries):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"}
        )

        text = response.choices[0].message.content

        try:
            return response_model.model_validate_json(text)
        except ValidationError as exc:
            print(f"[debug] Invalid JSON on attempt {attempt + 1}: {exc}")

    raise RuntimeError(f"Model did not return valid JSON after {retries} attempts.\nRaw output:\n{text}")
