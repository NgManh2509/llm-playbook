import os
from dotenv import load_dotenv
from groq import APIConnectionError, Groq, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()


class UnifiedLLMClient:
    def __init__(
        self,
        primary_model: str = "openai/gpt-oss-20b",
        fallback_model: str = "openai/gpt-oss-120b",
    ):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Chưa cấu hình GROQ_API_KEY trong file .env")
        self.client = Groq(api_key=api_key)
        self.primary_model = primary_model
        self.fallback_model = fallback_model

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    )
    def _call_groq(self, model: str, messages: list[dict], temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def generate(
        self,
        prompt: str,
        system_prompt: str = "Bạn là trợ lý AI hữu ích.",
        temperature: float = 0.7,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            return self._call_groq(self.primary_model, messages, temperature)
        except Exception as e:
            print(
                f"Primary model [{self.primary_model}] failed ({e}). "
                f"Fallback to [{self.fallback_model}]..."
            )
            return self._call_groq(self.fallback_model, messages, temperature)

