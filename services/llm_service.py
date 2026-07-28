from __future__ import annotations

import json
import re
from typing import Any

from groq import Groq
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import get_settings


class LLMService:
    """
    Central Groq LLM service.

    The service handles:
    - Normal text generation
    - JSON generation
    - Retry handling
    - JSON extraction and validation

    Counselling decisions are not hardcoded here.
    Those decisions will be made by the Counsellor Agent.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

        if not self.settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Add it to the .env file."
            )

        self.client = Groq(
            api_key=self.settings.groq_api_key,
        )

        self.model = self.settings.groq_model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4000,
    ) -> str:
        """
        Generate a normal text response.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
)

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        return content.strip()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(
            multiplier=2,
            min=2,
            max=20,
        ),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 3000,
    ) -> dict[str, Any]:
        """
        Ask Groq for a JSON object.

        This method avoids strict tool-call schema failures.
        The response is parsed and validated in Python.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n\n"
                        "Return only one valid JSON object. "
                        "Do not include markdown fences, "
                        "explanations, comments or extra text."
                    ),
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_object",
            },
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "Groq returned an empty JSON response."
            )

        return self._parse_json_object(content)

    @staticmethod
    def _parse_json_object(
        content: str,
    ) -> dict[str, Any]:
        """
        Parse a JSON object safely.

        Also handles responses accidentally wrapped in
        markdown code fences.
        """

        clean_content = content.strip()

        clean_content = re.sub(
            r"^```(?:json)?\s*",
            "",
            clean_content,
            flags=re.IGNORECASE,
        )

        clean_content = re.sub(
            r"\s*```$",
            "",
            clean_content,
        )

        try:
            parsed = json.loads(clean_content)
        except json.JSONDecodeError:
            start = clean_content.find("{")
            end = clean_content.rfind("}")

            if start == -1 or end == -1:
                raise ValueError(
                    "The LLM response does not contain "
                    "a valid JSON object."
                )

            parsed = json.loads(
                clean_content[start : end + 1]
            )

        if not isinstance(parsed, dict):
            raise ValueError(
                "Expected a JSON object from the LLM."
            )

        return parsed

    def health(self) -> dict[str, Any]:
        """
        Return LLM configuration information without
        exposing the API key.
        """

        return {
            "provider": "Groq",
            "model": self.model,
            "configured": bool(
                self.settings.groq_api_key
            ),
        }


_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """
    Return one shared LLM service instance.
    """

    global _llm_service

    if _llm_service is None:
        _llm_service = LLMService()

    return _llm_service