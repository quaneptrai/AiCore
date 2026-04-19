import os
import json
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pinecone import Pinecone
from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI
import uvicorn

app = FastAPI(title="Aris AI Matcher & Vector Engine")

# ================= 1. CẤU HÌNH & KHỞI TẠO =================
# Bạn có thể thay đổi API Key ở đây hoặc dùng file .env
PINECONE_API_KEY = "pcsk_6GmHeA_BpCsuvsQPoZhC6jgSJwwcnLhnEV4yX8boGxtEEuSkRP1fRH3exqPmEiRtP16YKG"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
INDEX_NAME = "thuctap"

# Khởi tạo Model Embedding (BGE-M3 hỗ trợ Hybrid & 1024 dim)
print("--- Đang tải BGE-M3 Model (Dense + Sparse)... ---")
embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Khởi tạo Pinecone & OpenAI
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
ai_client = OpenAI(api_key=OPENAI_API_KEY)

# ================= 2. DATA MODELS (DTOs) =================
class CVStructuredData(BaseModel):
    name: str
    education_degree: Optional[str]
    education_summary: Optional[str]
    skills: List[str]
    experience_summary: Optional[str]
    note: Optional[str]

class MatchRequest(BaseModel):
    text: str  # Nội dung JD

# ================= 3. CORE FUNCTIONS =================

def get_hybrid_embeddings(text: str):
    """
    Tạo ra Dense Vector (1024) và Sparse Vector cho Hybrid Search
    """
    output = embedding_model.encode([text], return_dense=True, return_sparse=True)
    
    # Dense vector (Phần Semantic - Ngữ nghĩa)
    dense_vec = output['dense_vecs'][0].tolist() # Dimension: 1024
    
    # Sparse vector (Phần Lexical - Từ khóa/BM25)
    sparse_raw = output['lexical_weights'][0]
    sparse_vec = {
        "indices": [int(k) for k in sparse_raw.keys()],
        "values": [float(v) for v in sparse_raw.values()]
    }
    return dense_vec, sparse_vec

# ================= 4. API ENDPOINTS =================

@app.post("/upload-cv-structured")
async def upload_cv_vector(data: CVStructuredData):
    """
    Nhận dữ liệu đã parse từ ASP.NET, tạo vector và đẩy lên Pinecone
    """
    try:
        # Gom nội dung để tạo vector "chất" nhất
        search_content = (
            f"Tên: {data.name}. "
            f"Bằng cấp: {data.education_degree}. "
            f"Học vấn: {data.education_summary}. "
            f"Kỹ năng: {', '.join(data.skills)}. "
            f"Kinh nghiệm: {data.experience_summary}. "
            f"Ghi chú: {data.note}"
        )

        dense, sparse = get_hybrid_embeddings(search_content)
        vector_id = str(uuid.uuid4())

        # Lưu vào Pinecone với Metadata chi tiết để lọc (Filtering)
        index.upsert([{
            "id": vector_id,
            "values": dense,
            "sparse_values": sparse,
            "metadata": {
                "type": "cv",
                "full_name": data.name,
                "skills": data.skills,
                "raw_text": search_content # Lưu text thô để LLM Rerank
            }
        }])

        return {"status": "success", "id": vector_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match-pro")
async def match_pro(request: MatchRequest):
    """
    Thực hiện Hybrid Search lấy Top 10, sau đó dùng LLM chấm điểm từng bản ghi
    """
    try:
        # 1. Tạo vector cho JD
        jd_dense, jd_sparse = get_hybrid_embeddings(request.text)

        # 2. Hybrid Search trên Pinecone
        search_results = index.query(
            vector=jd_dense,
            sparse_vector=jd_sparse,
            top_k=10, # Sàng lọc ra 10 ứng viên tiềm năng nhất
            include_metadata=True,
            filter={"type": {"$eq": "cv"}}
        )

        final_ranked_results = []

        # 3. Reranking & Scoring bằng LLM (Trí tuệ nhân tạo đọc hiểu)
        for match in search_results['matches']:
            cv_meta = match['metadata']
            
            # Prompt tối ưu để LLM chấm điểm khách quan
            prompt = f"""
            Bạn là một chuyên gia HR. Hãy chấm điểm mức độ phù hợp của ứng viên (0-100) dựa trên JD.
            
            JOB DESCRIPTION:
            {request.text}
            
            CV ỨNG VIÊN (Tóm tắt):
            {cv_meta['raw_text']}
            
            YÊU CẦU: Trả về JSON duy nhất: {{"score": <số>, "reason": "<lý do ngắn gọn>"}}
            """

            response = ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            
            ai_eval = json.loads(response.choices[0].message.content)

            final_ranked_results.append({
                "name": cv_meta['full_name'],
                "vector_similarity": round(match['score'], 4), # Điểm toán học
                "ai_matching_score": ai_eval.get("score"),    # Điểm đọc hiểu (Rerank)
                "explanation": ai_eval.get("reason"),
                "skills_found": cv_meta['skills']
            })

        # Sắp xếp lại danh sách dựa trên điểm của LLM (Chính xác hơn điểm Vector)
        final_ranked_results.sort(key=lambda x: x['ai_matching_score'], reverse=True)

        return {
            "total_found": len(final_ranked_results),
            "matches": final_ranked_results
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Matching Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)