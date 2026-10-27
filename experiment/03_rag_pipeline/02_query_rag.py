import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from typing import List
import httpx

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

load_dotenv()
client = Groq()
chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
collection = chroma_client.get_collection(
    name="policies_gemini",
    embedding_function=GeminiEmbeddingFunction(api_key=os.getenv("GEMINI_API_KEY"))
)

def retrive_context(query: str, top_k: int = 2) -> tuple[str, List[str]]:
    res = collection.query(n_results=top_k, query_texts=[query])

    context = []
    sources = []

    docs = res['documents'][0]
    metadatas = res['metadatas'][0]

    for i, (doc, metadatas) in enumerate(zip(docs, metadatas), start=1):
        section_name = metadatas.get("section", f"Mục {i}")
        context.append(f"[{i}] ({section_name}):\n{doc}")
        sources.append(section_name)

    context_str =  "\n\n".join(context)
    return context_str, sources

def ask(query: str) -> str:
    context_str, src = retrive_context(query)
    system_prompt = f"""Bạn là Trợ lý AI Chuyên viên Nhân sự nội bộ của công ty.
    Nhiệm vụ của bạn là giải đáp thắc mắc của nhân viên dựa trên các đoạn tài liệu được cung cấp dưới đây.
    TÀI LIỆU QUY CHẾ ĐƯỢC CẤP:
    --------------------------------------------------
    {context_str}
    --------------------------------------------------
    CÁC NGUYÊN TẮC BẮT BUỘC PHẢI TUÂN THỦ NGHIÊM NGẶT:
    1. CHỈ trả lời dựa trên thông tin CÓ THỰC trong tài liệu ở trên. Tuyệt đối KHÔNG dùng kiến thức bên ngoài, KHÔNG suy diễn hoặc giả định.
    2. BẮT BUỘC phải trích dẫn nguồn theo số [1], [2] ngay sau mỗi thông tin được lấy ra (Ví dụ: "Nhân viên được làm việc từ xa tối đa 2 ngày/tuần [1].").
    3. NẾU THÔNG TIN KHÔNG CÓ HOẶC KHÔNG ĐỦ TRONG TÀI LIỆU: Bắt buộc trả lời chính xác câu sau:
    "Thông tin này không được đề cập trong quy chế công ty."
    (Tuyệt đối cấm không được cố trả lời hoặc bịa ra con số).
    4. Giữ câu trả lời súc tích, chuyên nghiệp và đi thẳng vào trọng tâm."""

    res = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role" : "system", "content" : system_prompt},
            {"role" : "user", "content" : query}
        ],
        temperature=0.0
    )
    return res.choices[0].message.content


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    test_questions = [
        "Nhân viên thử việc có được hỗ trợ tiền gửi xe không?",
        "Quy định làm việc từ xa (WFH) thế nào và một năm tôi được nghỉ phép bao nhiêu ngày?",
        "Công ty có thưởng Tết âm lịch tháng lương thứ 13 không?",
    ]

    print(">>> BAT DAU CHAY THU NGHIEM HE THONG RAG (GROQ + CHROMADB):\n")
    for idx, q in enumerate(test_questions, start=1):
        print("=" * 70)
        print(f"[Cau hoi {idx}]: {q}")
        print("-" * 70)
        answer = ask(q)
        print(f"[Tra loi]:\n{answer}\n")
