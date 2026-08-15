from pathlib import Path
from pypdf import PdfReader

PDF_FOLDER = Path("data/raw_pdfs")
OUTPUT_FOLDER = Path("data/extracted")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

pdf_files = list(PDF_FOLDER.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files.")

for pdf_path in pdf_files:
    print(f"\nExtracting: {pdf_path.name}")

    reader = PdfReader(pdf_path)
    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    output_text = "\n\n".join(pages)

    output_file = OUTPUT_FOLDER / f"{pdf_path.stem}.txt"
    output_file.write_text(output_text, encoding="utf-8")

    print(f"Pages: {len(reader.pages)}")
    print(f"Characters extracted: {len(output_text):,}")
    print(f"Saved: {output_file}")

print("\nExtraction complete.")
