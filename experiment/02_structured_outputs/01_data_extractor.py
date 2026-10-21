import os
import json
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from groq import Groq

load_dotenv()
client = Groq()

RAW_CV = """
Chào anh/chị, em là Nguyễn Văn An, sinh năm 2000 ở Hà Nội. 
Em đã có 3 năm kinh nghiệm thực chiến làm lập trình viên Backend, thế mạnh lớn nhất là Python, FastAPI và PostgreSQL, ngoài ra có biết dùng Docker và Redis. 
Trước đây em từng làm ở công ty TechCorp tối ưu hệ thống chịu tải 100k CCU. 
Mức lương hiện tại của em là 25 triệu nhưng kỳ vọng ở vị trí mới là 35 triệu VNĐ/tháng.
Em có thể đi làm ngay từ đầu tháng sau.
"""

class CandidateProfile(BaseModel):
    full_name: str = Field(description="Họ và tên ứng viên")
    birth_year: int = Field(description="Năm sinh dạng số nguyên")
    years_of_experience: int = Field(description="Số năm kinh nghiệm làm việc")
    skills: list[str] = Field(description="Danh sách các công nghệ, kỹ năng")
    expected_salary_vnd: int = Field(description="Lương kỳ vọng đổi sang số nguyên VNĐ (VD: 35 triệu -> 35000000)")
    level: Literal["Fresher", "Junior", "Middle", "Senior"] = Field(description="Cấp độ kinh nghiệm")
    can_start_immediately: bool = Field(description="Có thể đi làm ngay hay không")

def extract_candidate_info(raw_text: str) -> CandidateProfile:
    schema_str = json.dumps(CandidateProfile.model_json_schema(), ensure_ascii=False)
    system_prompt = f"""
    Bạn là chuyên gia trích xuất dữ liệu tuyển dụng.
    Hãy phân tích văn bản của ứng viên và trả về kết quả dạng JSON tuân thủ nghiêm ngặt schema sau:
    {schema_str}
    """

    stream = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ], temperature=0.0,
        response_format={"type":"json_object"}
    )

    raw_json = stream.choices[0].message.content
    return CandidateProfile.model_validate_json(raw_json)

if __name__ == "__main__":
    print("⏳ Đang phân tích CV ứng viên...")
    profile = extract_candidate_info(RAW_CV)

    print("\n" + "=" * 50)
    print("🎉 KẾT QUẢ TRÍCH XUẤT (PYDANTIC OBJECT):")
    print("=" * 50)
    print(f"👤 Họ tên        : {profile.full_name}")
    print(f"📅 Năm sinh      : {profile.birth_year}")
    print(f"💼 Kinh nghiệm   : {profile.years_of_experience} năm")
    print(f"🏆 Cấp độ        : {profile.level}")
    print(f"🛠️ Kỹ năng       : {', '.join(profile.skills)}")
    print(f"💰 Lương kỳ vọng : {profile.expected_salary_vnd:,} VNĐ")
    print(f"🚀 Đi làm ngay   : {'Có' if profile.can_start_immediately else 'Không'}")

    print("\n🔍 KIỂM TRA ĐỘ CHUẨN XÁC VỀ KIỂU DỮ LIỆU:")
    print(f"- Type của profile: {type(profile)}")
    bonus = 5_000_000
    print(f"- Lương + Thưởng: {profile.expected_salary_vnd + bonus:,} VNĐ")
    print("\n📦 Dữ liệu JSON sẵn sàng gửi về Client / Database:")
    print(profile.model_dump_json(indent=2))

