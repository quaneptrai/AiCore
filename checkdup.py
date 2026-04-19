import json

with open("jobs.json", "r", encoding="utf-8") as f:
    jobs = json.load(f)

total = len(jobs)  # tổng số object

seen = set()
duplicates = []

for job in jobs:
    url = job.get("OriginalUrl")  # tránh crash nếu thiếu key
    if url in seen:
        duplicates.append(job)
    else:
        seen.add(url)

duplicate_count = len(duplicates)
unique_count = len(seen)

print(f"Tổng số object: {total}")
print(f"Số object trùng: {duplicate_count}")
print(f"Số object unique: {unique_count}")