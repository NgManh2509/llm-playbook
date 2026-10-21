import os
import json
import time
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from groq import Groq

load_dotenv()
client = Groq()

class IntentType(str, Enum):
    CHITCHAT = "CHITCHAT"
    RAG_SEARCH = "RAG_SEARCH"
    COMPLEX_REASONING = "COMPLEX_REASONING"

class RouteDecision(BaseModel):
    intent: IntentType = Field(description="Nhánh ý định phù hợp nhất")
    confidence: float = Field(description="Độ tin cậy từ 0.0 đến 1.0")
    reasoning: str = Field(description="Giải thích ngắn gọn trong 1 câu tại sao chọn nhánh này")

def route_intent(query: str) -> tuple[RouteDecision, float]:
    """
    Trả về:
    - RouteDecision: Quyết định phân loại của model
    - latency_ms: Thời gian thực thi (mili-giây)
    """
    schema_str = json.dumps(RouteDecision.model_json_schema(), ensure_ascii=False)
    system_prompt = f"""
    Bạn là bộ định tuyến ý định (Intent Router) chuyên nghiệp.
    Nhiệm vụ của bạn là đọc câu hỏi của người dùng và phân loại vào đúng 1 trong 3 nhóm sau:
    1. CHITCHAT: Chào hỏi, cảm ơn, nói chuyện phiếm thường ngày.
    2. RAG_SEARCH: Tra cứu thông tin, chính sách công ty, tài liệu, quy chế nội bộ.
    3. COMPLEX_REASONING: Viết code, thuật toán, giải toán, suy luận logic phức tạp.
    Trả về kết quả dưới dạng JSON tuân thủ nghiêm ngặt schema sau:
    {schema_str}
    """
    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    latency_ms = (time.perf_counter() - start_time)  * 1000
    decision = RouteDecision.model_validate_json(response.choices[0].message.content)
    return decision, latency_ms

TEST_QUERIES = [
    "Chào bot nhé, hôm nay thời tiết đẹp thật đấy!",
    "Quy chế nghỉ phép năm 2026 của công ty mình quy định tối đa bao nhiêu ngày một năm?",
    "Hãy viết cho tôi hàm đệ quy tính dãy Fibonacci trong Python và giải thích hiện tượng Stack Overflow.",
    "Công ty có hỗ trợ tiền gửi xe cho nhân viên thử việc không bạn?",
]
if __name__ == "__main__":
    print("🚦 BẮT ĐẦU KIỂM THỬ SEMANTIC INTENT ROUTER:\n")
    print("=" * 85)
    print(f"{'Câu hỏi':<45} | {'Intent':<18} | {'Độ trễ':<10} | {'Độ tin cậy'}")
    print("-" * 85)
    for q in TEST_QUERIES:
        decision, latency = route_intent(q)
        
        short_q = q if len(q) <= 42 else q[:39] + "..."
        print(f"{short_q:<45} | {decision.intent.value:<18} | {latency:>7.1f} ms | {decision.confidence*100:>5.1f}%")
        print(f"  ↳ Lý do: {decision.reasoning}\n")