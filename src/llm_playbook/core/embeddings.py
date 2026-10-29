import os
import chromadb
import httpx
from dotenv import load_dotenv

load_dotenv()


class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "models/gemini-embedding-2",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Chưa cấu hình GEMINI_API_KEY trong file .env")
        self.model_name = model_name
        self.url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"{model_name}:batchEmbedContents?key={self.api_key}"
        )

    def __call__(self, input: list[str]) -> list[list[float]]:
        payload = {
            "requests": [
                {
                    "model": self.model_name,
                    "content": {"parts": [{"text": text}]},
                }
                for text in input
            ]
        }
        res = httpx.post(self.url, json=payload, timeout=30.0).json()
        if "embeddings" not in res:
            raise ValueError(f"Lỗi từ Gemini API: {res}")
        return [item["values"] for item in res["embeddings"]]

