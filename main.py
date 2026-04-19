import os
import io
import re
import json
import uuid
import pyodbc
import pdfplumber
import tempfile
import logging
import asyncio
import httpx
import ollama
import torch
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import CrossEncoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    status,
    Body
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pinecone import Pinecone
from pdf2image import convert_from_path
from docling.document_converter import DocumentConverter
from rapidocr_onnxruntime import RapidOCR
from sentence_transformers import SentenceTransformer

# ================= CONFIG & LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ArisAI")

app = FastAPI(title="Aris AI Engine V7.2 - Local Embedding")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RECOMMEND_CACHE = {}
CACHE_TTL = 300

# ================= HARDWARE & MODELS CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🚀 Aris AI Engine đang chạy trên: {DEVICE}")
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=DEVICE
)
EMBED_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "vi": "dangvantuan/vietnamese-embedding",
    "mini": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "e5": "intfloat/multilingual-e5-large",
    "jina": "jinaai/jina-embeddings-v4",
}

loaded_models = {}

def get_local_model(model_key: str):
    m_key = model_key if model_key in EMBED_MODELS else "bge-m3"
    if m_key not in loaded_models:
        logger.info(f"📥 Đang tải model {m_key} vào {DEVICE}...")
        loaded_models[m_key] = SentenceTransformer(EMBED_MODELS[m_key], device=DEVICE)
    return loaded_models[m_key]

# ================= EXTERNAL CONFIG =================
HF_TOKEN = "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"
HF_CROSS_URL = "https://api-inference.huggingface.co/models/cross-encoder/ms-marco-MiniLM-L-6-v2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

PINECONE_KEY = "pcsk_4JWw7z_TY1XUsZDVAKwXpYEBiA1a9UCcraKD4MVmt9r9T56k1QNdzSo3Bgnep4bbgQTsW4"
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("thuctap")

ocr = RapidOCR()
doc_converter = DocumentConverter()

CATEGORIES_LIST = [
    "Công nghệ thông tin", "Kinh doanh & Bán hàng", "Marketing & Truyền thông",
    "Tài chính - Kế toán", "Hành chính - Nhân sự", "Y tế & Sức khỏe",
    "Giáo dục & Đào tạo", "Sản xuất & Kỹ thuật", "Thể hình & Thẩm mỹ",
    "Logistics & Xuất nhập khẩu", "Xây dựng & Kiến trúc", "Khách sạn & Nhà hàng",
    "Bất động sản", "Dịch vụ khách hàng", "Khác"
]

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ================= MODELS =================
class CVExtractData(BaseModel):
    name: str
    email: str
    category: str = Field(default="Khác")
    skills: Union[str, List[str]] 
    experiences: Union[str, List[Dict]]

class JobPostingRequest(BaseModel):
    Title: str
    Company: str
    Responsibilities: List[str]
    Requirements: List[str]
    Benefits: List[str]
    WorkTime: List[str]
    Salary: str
    Experience: str
    Deadline: Optional[str] = None
    cate: str
    keywords: List[str]
    Location_tags: List[str]
    Locations: List[str]
    FullText: str
    OriginalUrl: str

class EmbedJDRequest(BaseModel):
    jd_list: List[Dict[str, Any]]
    model: str = "bge-m3"

# ================= UTILS =================
def get_conn():
    conn_str = (
        "Driver={ODBC Driver 18 for SQL Server};"
        "Server=db47010.public.databaseasp.net;"
        "Database=db47010;"
        "UID=db47010;"
        "PWD=585810quan;"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)

async def get_embeddings(texts: List[str], model_key: str = "bge-m3") -> Optional[List[List[float]]]:
    try:
        model = get_local_model(model_key)
        embeddings = await asyncio.to_thread(
            model.encode, 
            texts, 
            convert_to_tensor=False, 
            show_progress_bar=False
        )
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"❌ Local Embedding error ({model_key}): {e}")
        return None

async def post_hf(url: str, payload: dict, retries: int = 3) -> Optional[Any]:
    async with httpx.AsyncClient(timeout=25.0) as client:
        for attempt in range(retries):
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (503, 429):
                    await asyncio.sleep(1.5 * (attempt + 1))
            except Exception as e:
                logger.warning(f"HF request failed (attempt {attempt+1}): {e}")
                await asyncio.sleep(1)
    return None

def safe_parse_json(text: str):
    try:
        clean = re.sub(r'```json|```', '', text).strip()
        match = re.search(r'(\[.*\]|\{.*\})', clean, re.DOTALL)
        return json.loads(match.group(0)) if match else None
    except:
        return None

def chunk_text(text: str, max_len: int = 700) -> List[str]:
    if not text: return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) <= max_len:
            current += (" " + s if current else s)
        else:
            if current.strip(): chunks.append(current.strip())
            current = s
    if current.strip(): chunks.append(current.strip())
    return chunks

async def extract_smart_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.split('.')[-1].lower()
    content = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name
    try:
        try:
            result = doc_converter.convert(temp_path)
            content = result.document.export_to_markdown()
        except: pass
        if len(content.strip()) < 300 and ext == "pdf":
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    content = "\n".join([p.extract_text() or "" for p in pdf.pages])
            except: pass
        if len(content.strip()) < 100:
            try:
                images = convert_from_path(temp_path)
                ocr_text = ""
                for img in images:
                    res, _ = ocr(img)
                    if res: ocr_text += "\n".join([line[1] for line in res])
                content = ocr_text
            except: pass
    finally:
        if os.path.exists(temp_path): os.unlink(temp_path)
    return content.strip()

# ================= SHARED VECTOR UPSERT =================
async def process_and_upsert_vector(
    base_id: str,
    text: str,
    category: str,
    vector_type: str,
    extra_metadata: dict,
    model_key: str = "bge-m3"
) -> bool:
    try:
        chunks = chunk_text(text)
        embeddings = await get_embeddings(chunks, model_key=model_key)

        if not embeddings: return False

        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            metadata = {
                "type": vector_type,
                "category": category,
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **extra_metadata
            }
            vectors.append({
                "id": f"{vector_type}_{base_id}_{i}",
                "values": emb,
                "metadata": metadata
            })

        if vectors:
            index.upsert(vectors=vectors)
            logger.info(f"✅ Upserted {len(vectors)} vectors | Type: {vector_type} | Model: {model_key}")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Upsert error: {e}")
        return False

# ================= BACKGROUND TASKS =================
async def background_process_cv(employee_id: int, base_id: str, data: CVExtractData, model_key: str):
    conn = None
    try:
        cv_emb_text = f"Category: {data.category}. Skills: {data.skills}. Experience: {data.experiences}"
        success = await process_and_upsert_vector(
            base_id=base_id,
            text=cv_emb_text,
            category=data.category,
            vector_type="cv",
            extra_metadata={"employee_id": employee_id},
            model_key=model_key
        )
        
        if success:
            conn = get_conn()
            cursor = conn.cursor()
            
            # Sửa lại câu lệnh MERGE khớp với Schema bảng Resumes của bạn
            # Thứ tự tham số trong VALUES: EmployeeId, Name, Email, SearchVectorContent
            sql_merge = """
                    MERGE INTO Resumes AS target
                    USING (VALUES (?, ?, ?, ?)) AS source (EmployeeId, Name, Email, SearchVectorContent)
                    ON target.EmployeeId = source.EmployeeId
                    WHEN MATCHED THEN 
                        UPDATE SET Name = source.Name, 
                                Email = source.Email, 
                                VectorId = ?, 
                                SearchVectorContent = source.SearchVectorContent,
                                Note = ISNULL(target.Note, N'Processed by AI'), 
                                LastUpdatedAt = GETUTCDATE()
                    WHEN NOT MATCHED THEN 
                        INSERT (EmployeeId, Name, Email, VectorId, SearchVectorContent, Note, CreatedAt, LastUpdatedAt)
                        VALUES (source.EmployeeId, source.Name, source.Email, ?, source.SearchVectorContent, N'New CV', GETUTCDATE(), GETUTCDATE());
                """
            
            # TRUYỀN THAM SỐ ĐÚNG THỨ TỰ:
            params = (
                employee_id,    # source.EmployeeId
                data.name,      # source.Name
                data.email,     # source.Email
                cv_emb_text,    # source.SearchVectorContent
                base_id,        # VectorId cho phần MATCHED
                base_id         # VectorId cho phần NOT MATCHED
            )
            
            cursor.execute(sql_merge, params)
            conn.commit()
            logger.info(f"✨ SQL DB updated for employee: {employee_id}")
    except Exception as e:
        logger.error(f"Background CV failed: {e}")
    finally:
        if conn: conn.close()

# ================= ENDPOINTS =================
@app.on_event("startup")
async def startup_event():
    get_local_model("bge-m3")

@app.post("/jobposting")
def create_jobposting(jobs: List[JobPostingRequest]):
    conn = None
    try:
        conn = get_conn()
        cursor = conn.cursor()
        inserted_count = 0
        for job in jobs:
            deadline_dt = None
            if job.Deadline:
                try: deadline_dt = datetime.strptime(job.Deadline, "%d/%m/%Y")
                except: pass

            sql_job = """
                INSERT INTO JobPostings 
                (Title, Company, Responsibilities, Requirements, Benefits, WorkTime, Salary, 
                Experience, Deadline, Category, Hashtags, LocationTags, FullText, OriginalUrl, 
                isActive, CreatedAt, UpdatedAt)
                OUTPUT INSERTED.Id
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, GETUTCDATE(), GETUTCDATE())
            """
            cursor.execute(sql_job, 
                job.Title, job.Company, 
                json.dumps(job.Responsibilities, ensure_ascii=False),
                json.dumps(job.Requirements, ensure_ascii=False),
                json.dumps(job.Benefits, ensure_ascii=False),
                ", ".join(job.WorkTime), job.Salary, job.Experience, 
                deadline_dt, job.cate,
                json.dumps(job.keywords, ensure_ascii=False),
                json.dumps(job.Location_tags, ensure_ascii=False),
                job.FullText, job.OriginalUrl
            )
            job_id = cursor.fetchone()[0]
            inserted_count += 1
            for loc_str in job.Locations:
                city = job.Location_tags[0] if job.Location_tags else ""
                address = loc_str
                if ":" in loc_str:
                    parts = loc_str.split(":", 1)
                    city, address = parts[0].strip(), parts[1].strip()
                cursor.execute("INSERT INTO JobLocations (JobPostingId, City, Address) VALUES (?, ?, ?)", job_id, city[:100], address[:2000])
        conn.commit()
        return {"status": "success", "message": f"Inserted {inserted_count} jobs"}
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(500, detail=str(e))
    finally:
        if conn: conn.close()

@app.post("/process-cv-full")
async def process_cv_full(
    employee_id: int = Form(...),
    file: UploadFile = File(...),
    model: str = Form("bge-m3"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")
    
    file_bytes = await file.read()
    full_text = await extract_smart_text(file_bytes, file.filename)
    if len(full_text.strip()) < 50:
        raise HTTPException(400, "Text extraction failed")

    prompt = f"Trích xuất JSON từ CV. Trường: name, email, category (chọn từ: {CATEGORIES_LIST}), skills, experiences.\nCV: {full_text[:4000]}"
    ollama_response = await asyncio.to_thread(ollama.chat, model="kimi-k2.5:cloud", 
                            messages=[{'role': 'system', 'content': 'Only return valid JSON.'}, {'role': 'user', 'content': prompt}])
    
    parsed = safe_parse_json(ollama_response['message']['content']) or {}
    cv_data = CVExtractData(**{**{"name": "Unknown", "email": "", "category": "Khác"}, **parsed})
    base_id = str(uuid.uuid4())
    
    background_tasks.add_task(background_process_cv, employee_id, base_id, cv_data, model)
    return {"status": "processing", "candidate": cv_data.name, "model": model, "vector_id": base_id}

@app.post("/embed-jd")
async def embed_jd(request: EmbedJDRequest): 
    model = request.model
    jd_list = request.jd_list
    
    if model not in EMBED_MODELS:
        raise HTTPException(status_code=400, detail="Model không hỗ trợ")

    try:
        count = 0
        for jd in jd_list:
            full_content = f"Category: {jd.get('cate')}\nTitle: {jd.get('Title')}\nRequirements: {jd.get('Requirements')}"
            base_id = str(uuid.uuid4())
            
            success = await process_and_upsert_vector(
                base_id=base_id,
                text=full_content,
                category=jd.get('cate', 'Khác'),
                vector_type="jd",
                extra_metadata={
                    "job_id": base_id, 
                    "title": jd.get('Title'), 
                    "company": jd.get('Company', '')
                },
                model_key=model
            )
            if success: count += 1
            
        return {"status": "success", "processed": count, "model_used": model}
    except Exception as e:
        logger.error(f"Embed JD error: {e}")
        raise HTTPException(500, detail=str(e))

@app.post("/recommend-jobs-for-employee", response_model=Dict[str, Any])
async def recommend_jobs_for_employee(
    employee_id: int = Form(...),
    top_k_vector: int = Form(80),
    top_k_final: int = Form(10)
):
    conn = None
    cache_key = f"rec_{employee_id}"

    try:
        now = time.time()
        if cache_key in RECOMMEND_CACHE:
            cached = RECOMMEND_CACHE[cache_key]
            if now - cached["time"] < CACHE_TTL:
                return cached["data"]

        conn = get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT r.VectorId, r.Name, r.Category, p.isStudent
            FROM Resumes r 
            JOIN Profiles p ON r.EmployeeId = p.Id
            WHERE r.EmployeeId = ?
        """, (employee_id,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="CV info not found in database")

        base_id, name, user_cate, is_student = row

        # ================= PINECODE FETCH (FIXED) =================
        target_id = f"cv_{base_id}_0"
        fetch_res = index.fetch(ids=[target_id])
        
        # Kiểm tra sự tồn tại của vector object chuẩn Pinecone SDK
        if not fetch_res or target_id not in fetch_res.vectors:
            # Trả về lỗi 404 sạch sẽ thay vì crash
            return JSONResponse(
                status_code=404,
                content={"message": "Dữ liệu Vector CV chưa sẵn sàng. Vui lòng thử lại sau vài giây."}
            )

        vec_data = fetch_res.vectors[target_id]
        query_vector = vec_data.values
        
        if not query_vector:
            raise HTTPException(status_code=404, detail="Vector values are empty")

        total_chunks = int(vec_data.metadata.get("total_chunks", 1))
        ids = [f"cv_{base_id}_{i}" for i in range(total_chunks)]

        fetch_all = index.fetch(ids=ids)
        
        all_chunks = []
        for c_id in ids:
            if c_id in fetch_all.vectors:
                v = fetch_all.vectors[c_id]
                all_chunks.append(v.metadata.get("text", ""))

        candidate_summary = " ".join(all_chunks)[:2000]

        # ================= SEARCH =================
        search_result = index.query(
            vector=query_vector,
            top_k=int(top_k_vector),
            include_metadata=True,
            filter={"type": "jd"} 
        )
        
        matches = search_result.matches[:30]
        if not matches:
            return {"status": "success", "candidate": name, "recommendations": []}

        # ================= CROSS ENCODER =================
        jd_texts = [
            (m.metadata.get("text", "")[:1200] if m.metadata else "") for m in matches
        ]


        pairs = [
            (candidate_summary[:1200], t)
            for t in jd_texts
        ]

        cross_scores = await asyncio.to_thread(
            cross_encoder.predict,
            pairs,
            batch_size=16,      
            show_progress_bar=False
        )

        ranked = []
        for i, match in enumerate(matches):
            meta = match.metadata if match.metadata else {}
            raw = 0.0
            try:
                if cross_scores and i < len(cross_scores):
                    cs = cross_scores[i]
                    if isinstance(cs, (int, float)): raw = float(cs)
                    elif isinstance(cs, dict) and "score" in cs: raw = float(cs["score"])
                    elif isinstance(cs, list) and len(cs) > 0:
                        raw = float(cs[0]["score"]) if isinstance(cs[0], dict) else float(cs[0])
            except: pass

            norm = max(0, min(100, raw * 20 + 50))
            ranked.append({
                "job_id": meta.get("job_id"),
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "jd_text": meta.get("text", ""),
                "location": meta.get("location", ""),
                "salary": meta.get("salary", ""),
                "exp": meta.get("experience", ""),
                "cross_score": round(norm, 2)
            })

        ranked = sorted(ranked, key=lambda x: x["cross_score"], reverse=True)
        seen, diversified = set(), []
        for job in ranked:
            key = (job["title"], job["company"])
            if key not in seen:
                diversified.append(job)
                seen.add(key)
            if len(diversified) >= 12: break

        # ================= LLM RERANK =================
        top_for_llm = diversified[:8]
        jobs_context = "\n".join([f"Job {i}: {j['title']} at {j['company']}" for i, j in enumerate(top_for_llm)])
        prompt = f"Candidate: {name}\nCV Summary: {candidate_summary}\nJobs:\n{jobs_context}\nReturn JSON array of objects with id, match_score, reason, advice."
        
        try:
            ai_res = await asyncio.to_thread(ollama.chat, model="kimi-k2.5:cloud", 
                                            messages=[{'role': 'user', 'content': prompt}])
            ai_list = safe_parse_json(ai_res['message']['content']) or []
        except: ai_list = []

        ai_map = {item.get("id"): item for item in ai_list if isinstance(item, dict)}

        final_results = []
        for idx, job in enumerate(top_for_llm):
            ai_data = ai_map.get(idx, {})
            llm_score = float(ai_data.get("match_score", 50))
            final_score = round((job["cross_score"] * 0.6) + (llm_score * 0.4), 2)
            final_results.append({
                "job_title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "salary": job["salary"],
                "final_score": final_score,
                "reason": ai_data.get("reason", ""),
                "advice": ai_data.get("advice", "")
            })

        result = {
            "status": "success",
            "candidate": name,
            "recommendations": sorted(final_results, key=lambda x: x["final_score"], reverse=True)[:int(top_k_final)]
        }
        RECOMMEND_CACHE[cache_key] = {"time": now, "data": result}
        return result

    except Exception as e:
        import traceback
        logger.error(f"Recommend error for employee {employee_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)