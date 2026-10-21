import os
import time
from typing import Generator
from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

class UnifiedLLMClient:
    def __init__(self,  primary_model: str = "openai/gpt-oss-20b", fallback_model: str = "openai/gpt-oss-120b"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.primary_model = primary_model
        self.fallback_model = fallback_model

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    )

    def _call_groq(self, model:str, mes: list, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(model=model, messages=mes, temperature=temperature)
        return response.choices[0].message.content
    
    def generate(self, prompt: str, system_prompt: str = "Bạn là trợ lý AI hữu ích.", temperature: float = 0.7 ) -> str:
        messages = [
            {
                "role": "system", "content" : system_prompt
            },
            {
                "role" : "user", "content": prompt
            }
        ]

        try:
            return self._call_groq(self.primary_model, messages, temperature)
        except Exception as e:
             print(f"Primary model [{self.primary_model}] failed ({e}). Fallback to [{self.fallback_model}]...")
             return self._call_groq(self.fallback_model, messages, temperature)

if __name__ == "__main__":
    print("--- 1. Test gọi bình thường ---")
    client = UnifiedLLMClient()
    res = client.generate("Tại sao bầu trời có màu xanh? Trả lời trong 1 câu ngắn.")
    print("Kết quả:", res)

    print("\n--- 2. Test kích hoạt Fallback tự động ---")
    broken_client = UnifiedLLMClient(primary_model="model-fake-khong-ton-tai")
    res_fallback = broken_client.generate("1 + 1 bằng mấy?")
    print("Kết quả:", res_fallback)

        