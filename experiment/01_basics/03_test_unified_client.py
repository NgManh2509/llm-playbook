from llm_playbook.core.llm_client import UnifiedLLMClient

if __name__ == "__main__":
    print("--- 1. Test gọi bình thường ---")
    client = UnifiedLLMClient()
    res = client.generate("Tại sao bầu trời có màu xanh? Trả lời trong 1 câu ngắn.")
    print("Kết quả:", res)

    print("\n--- 2. Test kích hoạt Fallback tự động ---")
    broken_client = UnifiedLLMClient(primary_model="model-fake-khong-ton-tai")
    res_fallback = broken_client.generate("1 + 1 bằng mấy?")
    print("Kết quả:", res_fallback)
