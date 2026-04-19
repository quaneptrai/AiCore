import json
import ollama
import os
from tqdm import tqdm

# --- CẤU HÌNH ---
MODEL_AI = "qwen2.5:1.5b"
INPUT_FILE = 'jobs.json'
OUTPUT_FILE = 'jobs_cate.json'

# Danh sách ngành nghề chuẩn để AI không phân loại lung tung
CATEGORIES_LIST = [
    "Công nghệ thông tin", "Kinh doanh & Bán hàng", "Marketing & Truyền thông",
    "Tài chính - Kế toán", "Hành chính - Nhân sự", "Y tế & Sức khỏe",
    "Giáo dục & Đào tạo", "Sản xuất & Kỹ thuật", "Thể hình & Thẩm mỹ",
    "Logistics & Xuất nhập khẩu", "Xây dựng & Kiến trúc", "Khách sạn & Nhà hàng",
    "Bất động sản", "Dịch vụ khách hàng", "Khác"
]

def process_and_categorize():
    # 1. Kiểm tra file đầu vào
    if not os.path.exists(INPUT_FILE):
        print(f" Lỗi: Không tìm thấy file {INPUT_FILE} trong thư mục hiện tại.")
        return

    # 2. Đọc toàn bộ file JSON
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        try:
            jobs = json.load(f)
        except json.JSONDecodeError:
            print(f" Lỗi: File {INPUT_FILE} không đúng định dạng JSON.")
            return

    print(f" Bắt đầu xử lý {len(jobs)} công việc...")

    # 3. Duyệt qua từng object trong file
    for job in tqdm(jobs, desc="Đang phân loại"):
        # Lấy dữ liệu để AI phân tích
        title = job.get("Title", "")
        # Kết hợp Responsibilities và Requirements (mỗi thứ lấy 2 dòng đầu cho nhẹ)
        resp = " ".join(job.get("Responsibilities", [])[:2])
        req = " ".join(job.get("Requirements", [])[:2])
        context = f"Title: {title}. Desc: {resp}. Req: {req}"

        # Prompt tối giản để AI trả về kết quả nhanh nhất
        prompt = f"""Phân loại công việc vào 1 nhóm từ danh sách: {CATEGORIES_LIST}
Chỉ trả về JSON định dạng: {{"cate": "tên_nhóm", "keywords": ["skill1", "skill2"]}}
Dữ liệu: {context}"""

        try:
            # Gọi Ollama chạy local
            response = ollama.generate(
                model=MODEL_AI, 
                prompt=prompt, 
                format='json', 
                options={'temperature': 0}
            )
            
            # Giải mã kết quả từ AI
            res_data = json.loads(response['response'])
            
            # Thêm trường mới vào object hiện tại
            job['cate'] = res_data.get("cate", "Khác")
            job['keywords'] = res_data.get("keywords", [])
            
        except Exception as e:
            # Nếu lỗi (ví dụ AI bị treo), gán giá trị mặc định để không dừng chương trình
            job['cate'] = "Khác"
            job['keywords'] = []

    # 4. Ghi toàn bộ dữ liệu đã xử lý ra file output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(jobs, f, ensure_ascii=False, indent=4)

    print(f"\nThanh cong! Đã lưu toàn bộ dữ liệu vào file: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_and_categorize()