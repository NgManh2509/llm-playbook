import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

PROMPT = "So sánh sự khác nhau giữa kiến trúc LPU (Language Processing Unit) và GPU khi xử lý suy luận LLM."

MODELS = [
    "openai/gpt-oss-20b", "openai/gpt-oss-120b"
]

def benchmark_model(model_name: str, prompt: str) -> dict:
    print(f"Benchmarking... {model_name}")
    start_time = time.perf_counter()
    first_token_time = None
    total_tokens = 0

    stream = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Bạn là trợ lý AI súc tích, trả lời bằng tiếng Việt."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        stream=True,
    )

    for chunk in stream:
        delta_content = chunk.choices[0].delta.content if chunk.choices else ""
        if first_token_time is None and delta_content:
            first_token_time = time.perf_counter()
        
        if hasattr(chunk, "x_groq") and chunk.x_groq and chunk.x_groq.usage:
            total_tokens = chunk.x_groq.usage.completion_tokens
    
    end_time = time.perf_counter()

    if first_token_time is None:
        first_token_time = end_time
    
    ttft_ms = (first_token_time - start_time) * 1000
    gen_time = end_time - first_token_time
    tps = total_tokens/gen_time if gen_time > 0 else 0
    total_time_s = end_time - start_time

    return {
        "model": model_name,
        "ttft_ms": ttft_ms,
        "tps": tps,
        "tokens": total_tokens,
        "total_time_s": total_time_s,
    }

if __name__ == "__main__":
    results = []
    for model in MODELS:
        res = benchmark_model(model, PROMPT)
        results.append(res)
    print("\n" + "=" * 75)
    print(f"{'Model':<25} | {'TTFT (ms)':<12} | {'TPS (tokens/s)':<16} | {'Tokens':<8} | {'Tổng (s)'}")
    print("-" * 75)
    for r in results:
        print(f"{r['model']:<25} | {r['ttft_ms']:<12.1f} | {r['tps']:<16.1f} | {r['tokens']:<8} | {r['total_time_s']:.2f}s")
    print("=" * 75)
