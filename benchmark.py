import json
import time
import psutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

HF_TOKEN = "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"
login(token=HF_TOKEN)

# =========================
# MODELS
# =========================
MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "e5-large": "intfloat/multilingual-e5-large",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2"
}

MODEL_CACHE = {}

# =========================
# LOAD DATA (JSON LIST)
# =========================
def load_jobs(file_path, n=5):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # lấy random thay vì slice
    import random
    random.shuffle(data)

    vn_jobs = data[:n]
    global_jobs = data[n:n*2]

    return vn_jobs, global_jobs


# =========================
# TEXT NORMALIZATION
# =========================
def job_to_text(job):
    return " | ".join([
        job.get("Title", ""),
        job.get("Company", ""),
        job.get("cate", ""),
        job.get("Salary", ""),
        job.get("Experience", ""),
        " ".join(job.get("Location_tags", [])),
        " ".join(job.get("Responsibilities", [])),
        " ".join(job.get("Requirements", [])),
        " ".join(job.get("keywords", [])),
        job.get("FullText", "")
    ])


# =========================
# MODEL LOADER (CACHE)
# =========================
def get_model(model_name):
    if model_name not in MODEL_CACHE:
        MODEL_CACHE[model_name] = SentenceTransformer(MODELS[model_name])
    return MODEL_CACHE[model_name]


# =========================
# EMBEDDING + SPEED
# =========================
def embed(model_name, texts):
    model = get_model(model_name)

    start = time.time()
    vectors = model.encode(texts, normalize_embeddings=True)
    end = time.time()

    speed = len(texts) / (end - start + 1e-9)
    return vectors, speed


# =========================
# METRICS
# =========================
def mean_similarity(vecs):
    sim = cosine_similarity(vecs)
    n = len(vecs)

    total = 0
    count = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                total += sim[i][j]
                count += 1

    return total / count


# 🔥 retrieval-style cross similarity (IMPORTANT FIX)
def cross_similarity(a, b):
    sim = cosine_similarity(a, b)
    return np.mean(np.max(sim, axis=1))


def quality(vn, gl, cross):
    return (vn + gl) - cross


def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


# =========================
# BENCHMARK CORE
# =========================
def benchmark(vn_jobs, global_jobs):
    results = []

    vn_texts = [job_to_text(j) for j in vn_jobs]
    global_texts = [job_to_text(j) for j in global_jobs]

    for name, model_path in MODELS.items():
        print(f"\n🚀 Running model: {name}")

        try:
            mem_before = get_memory_mb()

            # EMBED
            vn_vecs, speed_vn = embed(name, vn_texts)
            gl_vecs, speed_gl = embed(name, global_texts)

            mem_after = get_memory_mb()

            # METRICS
            vn_score = mean_similarity(vn_vecs)
            gl_score = mean_similarity(gl_vecs)
            cross = cross_similarity(vn_vecs, gl_vecs)

            qual = quality(vn_score, gl_score, cross)
            speed = (speed_vn + speed_gl) / 2
            memory = mem_after - mem_before
            dim = len(vn_vecs[0])

            # FINAL SCORE (balanced)
            final_score = (
                qual * 0.6 +
                speed * 0.2 +
                (1 / (memory + 1e-6)) * 0.1 +
                (1 if dim >= 1024 else 0.5) * 0.1
            )

            results.append({
                "model": name,
                "dim": dim,
                "vn_score": vn_score,
                "global_score": gl_score,
                "cross_similarity": cross,
                "quality": qual,
                "speed_docs_per_sec": speed,
                "memory_mb": memory,
                "final_score": final_score
            })

        except Exception as e:
            print(f"❌ Error {name}: {e}")

    return sorted(results, key=lambda x: x["final_score"], reverse=True)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    vn_jobs, global_jobs = load_jobs("jobs_cate.json", n=5)

    results = benchmark(vn_jobs, global_jobs)

    print("\n\n==================== FINAL RANKING ====================\n")

    for r in results:
        print(f"""
Model: {r['model']}
Dim: {r['dim']}
VN Score: {r['vn_score']:.4f}
Global Score: {r['global_score']:.4f}
Cross Similarity: {r['cross_similarity']:.4f}
Accuracy Proxy: {r['quality']:.4f}
Speed (docs/s): {r['speed_docs_per_sec']:.2f}
Memory (MB): {r['memory_mb']:.2f}
FINAL SCORE: {r['final_score']:.4f}
--------------------------------------------------
""")