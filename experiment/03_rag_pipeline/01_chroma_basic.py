import os
import httpx
import chromadb
from dotenv import load_dotenv

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-2"):
        self.api_key = api_key
        self.model_name = model_name
        self.url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:batchEmbedContents?key={self.api_key}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        payload = {
            "requests": [
                {
                    "model": self.model_name,
                    "content": {
                        "parts": [{"text": text}]
                    }
                }
                for text in input
            ]
        }
        res = httpx.post(self.url, json=payload, timeout=30.0).json()
        if "embeddings" not in res:
            raise ValueError(f"Lỗi từ Gemini API: {res}")
        return [item["values"] for item in res["embeddings"]]

chroma_client = chromadb.PersistentClient(path="./data/chroma_db")

collection = chroma_client.get_or_create_collection(
    name="policies_gemini",
    embedding_function=GeminiEmbeddingFunction(api_key=GEMINI_KEY)
)

file_path = "data/company_policy.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

raw_sections = content.split("## ")
documents = []
ids = []
metadatas = []

for idx, section in enumerate(raw_sections):
    clean_text = section.strip()
    if not clean_text or clean_text.startswith("# QUY CHẾ"):
        continue

    first_line = clean_text.split("\n")[0].strip()
    documents.append(clean_text)
    ids.append(f"chunk_{idx}")
    metadatas.append({"source": "company_policy.md", "section": first_line})

collection.upsert(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print(f"Đã nạp thành công {collection.count()} chunks vào ChromaDB với Gemini Vector!\n")

query = "Nhân viên thử việc có được hỗ trợ tiền gửi xe không?"
res = collection.query(query_texts=[query], n_results=1)

print(f"Câu hỏi: {query}")
print(f"Đoạn trích tìm thấy trong tài liệu:")
print(f"   ↳ {res['documents'][0][0]}")
print(f"Nguồn trích dẫn: {res['metadatas'][0][0]['section']}")
