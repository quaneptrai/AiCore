import pdfplumber
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions # Sửa lỗi import ở đây
from rapidocr_onnxruntime import RapidOCR
from pdf2image import convert_from_path
import tkinter as tk
from tkinter import filedialog
import os

# =========================
# INIT OCR
# =========================
ocr = RapidOCR()

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn CV PDF",
        filetypes=[("PDF files", "*.pdf")]
    )
    return file_path

# =========================
# 1. DOCLING EXTRACT (Ưu tiên số 1)
# =========================
def extract_with_docling(file_path):
    try:
        # Cấu hình chuẩn cho phiên bản Docling mới
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False 
        
        # Khởi tạo converter với pipeline_options
        converter = DocumentConverter() 
        # Lưu ý: Một số bản Docling dùng tham số khác, nếu dòng dưới lỗi 
        # hãy thử: converter = DocumentConverter(pipeline_options=pipeline_options)
        
        result = converter.convert(file_path)
        return result.document.export_to_markdown()
    except Exception as e:
        print(f"⚠️ Docling lỗi: {e}")
        return None

# =========================
# 2. PDFPLUMBER (Dự phòng văn bản số)
# =========================
def extract_with_pdfplumber(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text(layout=True)
                if t:
                    text += t + "\n"
        return text.strip() if len(text.strip()) > 100 else None
    except Exception as e:
        print(f"⚠️ PDFPlumber lỗi: {e}")
        return None

# =========================
# 3. RAPID OCR (Dự phòng cuối cùng)
# =========================
def extract_with_ocr(file_path):
    print("⚠️ Đang quét OCR (File scan hoặc ảnh)...")
    try:
        images = convert_from_path(file_path)
        text = ""
        for img in images:
            result, _ = ocr(img)
            if result:
                for line in result:
                    text += line[1] + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ OCR thất bại hoàn toàn: {e}")
        return ""

# =========================
# MAIN PIPELINE
# =========================
def get_full_text(file_path):
    if not file_path:
        return "Chưa chọn file."

    print(f"\n📄 Đang xử lý: {os.path.basename(file_path)}")

    # Thử Docling
    full_text = extract_with_docling(file_path)
    method = "Docling (Markdown)"

    # Fallback 1: PDFPlumber
    if not full_text or len(full_text) < 200:
        full_text = extract_with_pdfplumber(file_path)
        method = "PDFPlumber (Raw Text)"

    # Fallback 2: OCR
    if not full_text or len(full_text) < 50:
        full_text = extract_with_ocr(file_path)
        method = "RapidOCR (Scan)"

    return {
        "method": method,
        "content": full_text
    }

if __name__ == "__main__":
    path = select_file()
    if path:
        result = get_full_text(path)
        print("\n" + "="*30)
        print(f"PHƯƠNG PHÁP TRÍCH XUẤT: {result['method']}")
        print("="*30)
        print(result['content'])
    else:
        print("❌ Bạn chưa chọn file.")