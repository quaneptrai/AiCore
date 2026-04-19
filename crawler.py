# -*- coding: utf-8 -*-
import os
import re
import time
import json
import random
import pyodbc
import datetime
import warnings
import cloudscraper
import threading
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
# ================= CẤU HÌNH HỆ THỐNG =================
warnings.filterwarnings("ignore")
MAX_WORKERS = 5      
SELENIUM_TIMEOUT = 10  
BATCH_SIZE = random.randint(3, 6)   

RANDOM_SLEEP = (3, 7)
progress_counter = 0
progress_lock = threading.Lock()
seen_lock = threading.Lock()
DRIVER_BUSY = {}
DRIVER_LOCK = threading.Lock()
DRIVER_POOL = []
results = []
USER_AGENTS_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/122.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0"
]

# ================= TIỆN ÍCH (UTILS) =================
def clean_url(url):
    """Cắt bỏ toàn bộ tham số phía sau dấu ? để tránh trùng lặp 1 job thành 2 link"""
    if not url: return ""
    return url.split('?')[0].strip()

def is_active(deadline):
    """Kiểm tra xem Job còn hạn không"""
    if not deadline: return True
    try:
        clean_date = re.search(r"(\d{2}/\d{2}/\d{4})", deadline)
        if clean_date:
            deadline_date = datetime.datetime.strptime(clean_date.group(1), "%d/%m/%Y")
            return deadline_date >= datetime.datetime.now()
    except: pass
    return True

def init_driver_pool(size=3):
    for i in range(size):
        ua = random.choice(USER_AGENTS_POOL)
        driver = create_driver(ua)
        DRIVER_POOL.append(driver)
        DRIVER_BUSY[driver] = False

def get_driver():
    while True:
        with DRIVER_LOCK:
            for d in DRIVER_POOL:
                if not DRIVER_BUSY[d]:
                    DRIVER_BUSY[d] = True
                    return d
        time.sleep(0.5)    
def release_driver(driver):
    with DRIVER_LOCK:
        DRIVER_BUSY[driver] = False
        
# ================= KIỂM TRA DỮ LIỆU CŨ =================
def get_existing_urls():
    """Lấy danh sách link ĐÃ CÓ từ cả SQL Database và File JSON"""
    urls = set()
    
    try:
        conn = pyodbc.connect(
            "Driver={ODBC Driver 18 for SQL Server};Server=db47010.public.databaseasp.net;"
            "Database=db47010;UID=db47010;PWD=585810quan;Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=5;"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT OriginalUrl FROM JobPostings")
        for row in cursor.fetchall():
            urls.add(clean_url(row[0]))
        conn.close()
    except Exception as e:
        print(f" Không thể kết nối DB (Bỏ qua bước check DB): {e}")

    if os.path.exists("jobs.json"):
        try:
            with open("jobs.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    if 'OriginalUrl' in item:
                        urls.add(clean_url(item['OriginalUrl']))
        except Exception as e:
            print(f"Lỗi đọc jobs.json: {e}")
            
    return urls

# ================= BỘ PHÂN TÍCH DỮ LIỆU (PARSER) =================
def parse_job_from_soup(soup, url):
    """Bóc tách chi tiết: Fix Location, Thêm WorkTime, Xóa Bullet points"""
    job_section = soup.select_one("div.job-description")
    if not job_section: return None

    data = {
        "OriginalUrl": clean_url(url),
        "URL": clean_url(url),
        "Title": soup.select_one("h1").get_text(strip=True) if soup.select_one("h1") else "N/A",
        "Company": soup.select_one("div.company-name-label a").get_text(strip=True) if soup.select_one("div.company-name-label a") else "N/A",
        "Responsibilities": [], "Requirements": [], "Benefits": [],
        "Locations": [],"Location_tags": [], "WorkTime": [], "Salary": "Thỏa thuận", "Deadline": "", "Experience": "N/A"
    }

    # --- 1. Tiện ích dọn dẹp Bullet Points ---
    def clean_bullet(text):
        if not text: return ""
        # Xóa các ký tự gạch đầu dòng -, *, + và khoảng trắng dư thừa
        return re.sub(r"^[-\*\+\s•]+", "", text).strip()

    # --- 2. Lấy Salary ,Location Tags, Experience từ Header ---
    header_container = soup.select_one(".job-detail__info--sections")
    if header_container:
        # 1. Bóc tách Mức lương
        salary_sec = header_container.select_one(".section-salary .job-detail__info--section-content-value")
        if salary_sec:
            data["Salary"] = salary_sec.get_text(strip=True)

        # 2. Bóc tách location_tags (Hà Nội, TP.HCM...)
        location_sec = header_container.select_one(".section-location .job-detail__info--section-content-value")
        if location_sec:
            # Lấy text từ thẻ <a> hoặc <span> bên trong giá trị
            tags = [a.get_text(strip=True) for a in location_sec.select("a, span") if a.get_text(strip=True)]
            # Nếu không có thẻ con, lấy text trực tiếp
            data["Location_tags"] = list(set(tags)) if tags else [location_sec.get_text(strip=True)]

        # 3. Bóc tách Kinh nghiệm
        exp_sec = header_container.select_one(".section-experience .job-detail__info--section-content-value")
        if exp_sec:
            data["Experience"] = exp_sec.get_text(strip=True)

    # --- 3. Quét các khối job-description__item dựa trên H3 ---
    for item in job_section.select("div.job-description__item"):
        h3 = item.select_one("h3")
        if not h3: continue
        
        title = h3.get_text(strip=True).lower()
        content_box = item.select_one(".job-description__item--content")
        if not content_box: continue
        
        # Lấy danh sách các dòng text và dọn dẹp bullet
        raw_texts = [x.get_text(strip=True) for x in content_box.select("p, li, div") if x.get_text(strip=True)]
        clean_texts = [clean_bullet(t) for t in raw_texts if len(clean_bullet(t)) > 2]

        if "mô tả" in title: 
            data["Responsibilities"] = clean_texts
        elif "yêu cầu" in title: 
            data["Requirements"] = clean_texts
        elif "quyền lợi" in title: 
            data["Benefits"] = clean_texts
        elif "địa điểm" in title:
            # Lọc bỏ dòng chú thích "đã được cập nhật..."
            data["Locations"] = [clean_bullet(t) for t in raw_texts if "đã được cập nhật" not in t]
        elif "thời gian" in title:
            data["WorkTime"] = clean_texts

    # --- 4. Bóc tách Deadline ---
    dl = soup.select_one(".job-detail__info--deadline")
    if dl: 
        data["Deadline"] = re.sub(r"Hạn nộp hồ sơ:\s*|\(Còn\s*\d+\s*ngày\)", "", dl.get_text(strip=True), flags=re.IGNORECASE).strip()

    # Gom nhóm dữ liệu cho AI (RAG)
    data["FullText"] = (
        f"Title: {data['Title']}\nCompany: {data['Company']}\n"
        f"Exp: {data['Experience']} | Salary: {data['Salary']}\n"
        f"WorkTime: {', '.join(data['WorkTime'])}\n"
        f"Locations: {', '.join(data['Locations'])}\n"
        f"Mô tả: {' '.join(data['Responsibilities'])}"
    )
    data["isActive"] = is_active(data["Deadline"])
    
    return data


def simulate_human_behavior(driver):
    """Giả lập hành vi: cuộn trang zig-zag, dừng đọc, di chuyển chuột"""
    try:
        total_height = driver.execute_script("return document.body.scrollHeight")
        current_pos = 0
        while current_pos < total_height * 0.6: 
            step = random.randint(250, 500)
            current_pos += step
            driver.execute_script(f"window.scrollTo(0, {current_pos});")
            time.sleep(random.uniform(0.8, 2.0)) 
            
            if random.random() > 0.7:
                driver.execute_script(f"window.scrollBy(0, -150);")
                time.sleep(0.5)

        from selenium.webdriver.common.action_chains import ActionChains
        elements = driver.find_elements(By.CSS_SELECTOR, "h3, .job-description__item")
        if elements:
            action = ActionChains(driver)
            target = random.choice(elements[:3])
            action.move_to_element(target).pause(random.uniform(0.5, 1.5)).perform()

    except: pass


# ================= CÁC PHA CRAWL =================
def phase1_fast(url):
    """Phase 1: Giả lập request từ trình duyệt thật với Referer ngẫu nhiên"""
    referers = [
        "https://www.google.com/",
        "https://www.topcv.vn/",
        "https://www.topcv.vn/viec-lam",
        "https://www.facebook.com/"
    ]
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
    )
    time.sleep(random.uniform(2, 4))
    
    headers = {
        'Referer': random.choice(referers),
        'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }
    
    try:
        res = scraper.get(url, headers=headers, timeout=10)
        if res.status_code == 200 and "job-description" in res.text:
            return parse_job_from_soup(BeautifulSoup(res.text, "html.parser"), url)
    except: pass
    return None

def create_driver(ua):
    """Tạo trình duyệt ẩn cho Selenium"""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument(f"user-agent={ua}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--blink-settings=imagesEnabled=false")
    options.add_argument("--log-level=3") 
    return webdriver.Chrome(options=options)

def phase2_retry(url):
    for i in range(1, 4):
        driver = get_driver()
        try:
            driver.get("https://www.google.com")
            time.sleep(random.uniform(1, 2))

            driver.execute_script(f"window.open('{url}', '_blank');")
            driver.switch_to.window(driver.window_handles[-1])

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "job-description"))
            )

            simulate_human_behavior(driver)

            data = parse_job_from_soup(
                BeautifulSoup(driver.page_source, "html.parser"), url
            )

            driver.close()
            driver.switch_to.window(driver.window_handles[0])

            release_driver(driver)

            if data:
                return data, "Phase 2 OK"

        except:
            try:
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
            except:
                pass

            release_driver(driver)

        time.sleep(random.uniform(3, 6))

    return None, "Fail"

def auto_worker(link, total_links):
    global progress_counter
    time.sleep(random.uniform(2, 8))
    
    data = phase1_fast(link)
    status = "Phase 1 (Nhanh)"

    if not data:
        if random.random() < 0.3:
            time.sleep(random.uniform(6, 10))
        else:
            time.sleep(random.uniform(2, 5))
        data, status = phase2_retry(link)

    with progress_lock:
        progress_counter += 1
        curr = progress_counter
    
    icon = "✅" if data else "❌"
    print(f"{icon} [{curr}/{total_links}] {status} -> {link}")
    return data

# ================= HÀM MAIN =================
if __name__ == "__main__":
    existing_urls = get_existing_urls()
    seen_urls = set(existing_urls)
    options = uc.ChromeOptions()
    driver_main = uc.Chrome(options=options)
    init_driver_pool(size=3)
    driver_main.get("https://www.topcv.vn/viec-lam")
    
    print("Bây giờ hãy chọn Filter trên trình duyệt.")
    input("Khi danh sách Job đã hiện ra, bấm Enter tại đây để lấy link...")
    
    try: pages_to_crawl = int(input("Bạn muốn quét bao nhiêu trang? (Ví dụ: 3): ") or 1)
    except: pages_to_crawl = 1
    
    raw_links = []
    print(f"\nĐang thu thập link từ {pages_to_crawl} trang...")
    for p in range(pages_to_crawl):
        print(f"   Đang lấy trang {p+1}...")
        try:
            WebDriverWait(driver_main, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".feature-job-item h3 a")))
            jobs = driver_main.find_elements(By.CSS_SELECTOR, ".feature-job-item h3 a")
            raw_links.extend([j.get_attribute("href") for j in jobs])
            
            if p < pages_to_crawl - 1:
                next_btn = driver_main.find_element(By.CSS_SELECTOR, ".btn-feature-jobs-next")
                driver_main.execute_script("arguments[0].click();", next_btn)
                time.sleep(3)
        except Exception as e:
            print("Không tìm thấy trang tiếp theo hoặc lỗi:", e)
            break
            
    driver_main.quit()
    
    
    # Chỉ giữ lại các link thực sự mới (đã loại bỏ dấu ?)
    links_to_crawl = list(set([l for l in raw_links if clean_url(l) not in existing_urls]))
    
    print(f"📊 Tổng link thu được: {len(list(set(raw_links)))}")
    print(f"🎯 Số link MỚI cần crawl: {len(links_to_crawl)}\n")

    if len(links_to_crawl) == 0:
        print("☕ Không có Job nào mới. Mọi thứ đã được cập nhật!")
        exit()

    results = []
    failed_links = []
    total_links = len(links_to_crawl)
    start_time = time.time()

    for i in range(0, total_links, BATCH_SIZE):
        batch = links_to_crawl[i : i + BATCH_SIZE]
        print(f"\n🚀 Đang xử lý Batch {i//BATCH_SIZE + 1} ({len(batch)} jobs)...")
        
        # Chạy đa luồng cho 5 link trong batch này
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(auto_worker, link, total_links): link for link in batch}
            for future in as_completed(futures):
                link = futures[future]   
                res = future.result()

                if res:
                    u = clean_url(res['OriginalUrl'])
                    with seen_lock:
                        if u not in seen_urls:
                            results.append(res)
                            seen_urls.add(u)
                else:
                    failed_links.append(link)
        if i + BATCH_SIZE < total_links:
            if random.random() < 0.2:
                break_time = random.randint(15, 25)
            else:
                break_time = random.randint(5, 10)

            print(f"⏳ Nghỉ {break_time}s...")
            time.sleep(break_time)
        if (i // BATCH_SIZE + 1) % 5 == 0:
            print("♻️ Reset driver pool...")
            for d in DRIVER_POOL:
                try: d.quit()
                except: pass
            DRIVER_POOL.clear()
            DRIVER_BUSY.clear()
            init_driver_pool(size=3)
    if failed_links:
        print(f"\n🔁 Retry {len(failed_links)} link fail...")

        for link in failed_links:
            data, _ = phase2_retry(link)
            if data:
                u = clean_url(data['OriginalUrl'])
                with seen_lock:
                    if u not in seen_urls:
                        results.append(data)
                        seen_urls.add(u)
    # --- LƯU JSON AN TOÀN ---
    if results:
        print("\n💾 Đang tiến hành lưu File ...")
        
        current_file_data = []
        if os.path.exists("jobs.json"):
            with open("jobs.json", "r", encoding="utf-8") as f:
                try: current_file_data = json.load(f)
                except: current_file_data = []

        existing_map = {clean_url(item['OriginalUrl']): True for item in current_file_data if 'OriginalUrl' in item}

        new_count = 0
        for r in results:
            u = clean_url(r['OriginalUrl'])
            if u not in existing_map:
                current_file_data.append(r)
                existing_map[u] = True
                new_count += 1

        with open("jobs.json", "w", encoding="utf-8") as f:
            json.dump(current_file_data, f, ensure_ascii=False, indent=4)
            
        time_taken = round(time.time() - start_time, 2)
        print(f"✅ Hoàn thành trong {time_taken} giây!")
        print(f"✨ Đã thêm {new_count} job mới. Tổng cộng: {len(current_file_data)} Jobs.")
    else:
        print("\n❌ Không lấy được dữ liệu nào thành công.")