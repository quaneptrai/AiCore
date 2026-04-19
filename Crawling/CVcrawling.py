import pdfplumber

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {i+1} ---\n"
                text += page_text
    return text

if __name__ == "__main__":
    pdf_path = "cv.pdf"  # đổi thành file của bạn
    content = extract_text_from_pdf(pdf_path)

    print(content[:1000])  # in thử 1000 ký tự