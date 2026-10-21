import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Không tìm thấy API key trong .env")

client = Groq(api_key=api_key)

model_name = "openai/gpt-oss-20b"

prompt = "Giải thích ngắn gọn trong 3 câu: Tại sao chip LPU của Groq lại suy luận LLM nhanh hơn GPU truyền thống?"

print(f"Đang gọi {model_name} : ")
start_time = time.time()

stream = client.chat.completions.create(model=model_name, messages=[
        {"role": "system", "content": "Bạn là trợ lý AI súc tích, trả lời bằng tiếng Việt."},
        {"role": "user", "content": prompt},
    ], temperature=0.7, stream=True)

first_token = True
for chunk in stream:
    content = chunk.choices[0].delta.content or ""
    if first_token and content:
        ttft = time.time() - start_time
        print(f"⚡ [Time-to-First-Token: {ttft*1000:.1f}ms]\n")
        first_token = False
    print(content, end="", flush=True)

total_time = time.time() - start_time
print(f"\n⚡ [Total time: {total_time*1000:.1f}ms]")
