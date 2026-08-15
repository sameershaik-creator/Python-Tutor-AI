import re
from pathlib import Path

EXTRACTED_FOLDER = Path("data/extracted")
CLEANED_FOLDER = Path("data/cleaned")

CLEANED_FOLDER.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    # 1. Fix common OCR mistakes specifically found in the handwriting notes
    ocr_fixes = {
        "brint": "print",
        "Vexy Simple Syntax": "Very Simple Syntax",
        "Assignmen Operataxs": "Assignment Operators",
        "Janguage": "Language",
        "Fasy": "Easy",
        "interpretem": "interpreter",
        "Bynamically": "Dynamically",
        "fuom": "from",
        "Theyaче": "They are",
        "thete variable": "their variable",
        "realising": "releasing",
        "Дморвох": "Dropbox",
        "operakors": "operators",
        "yaiable": "variable",
        "artbnmetic": "arithmetic",
        "loqica": "logical",
        "Boalean": "Boolean",
        "tepnsenhaians": "representations",
        "Membesshbip": "Membership",
        "dest whether": "test whether",
        "entity Opetators": "Identity Operators",
    }
    
    for wrong, right in ocr_fixes.items():
        text = text.replace(wrong, right)
        
    # 2. Remove typical page headers/footers/watermarks in handwritten notes
    watermarks = [
        "classMAte",
        "Copyrighted by CodelithCuvious ·Com",
        "Spiral",
        "Date",
        "Page"
    ]
    for wm in watermarks:
        # Using regex to remove them safely when they appear alone on a line
        text = re.sub(rf'^\s*{wm}\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(rf'^\s*{wm}.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # 3. Clean up white spaces
    # Remove leading/trailing spaces on each line
    text = '\n'.join(line.strip() for line in text.splitlines())
    
    # Remove excessive blank lines (more than 2 down to 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def main():
    txt_files = list(EXTRACTED_FOLDER.glob("*.txt"))
    print(f"Found {len(txt_files)} extracted files to clean.\n")
    
    for file in txt_files:
        print(f"Cleaning: {file.name}")
        raw_text = file.read_text(encoding="utf-8", errors="replace")
        
        cleaned_text = clean_text(raw_text)
        
        output_file = CLEANED_FOLDER / file.name
        output_file.write_text(cleaned_text, encoding="utf-8")
        
        # Show a brief diff in size
        orig_len = len(raw_text)
        new_len = len(cleaned_text)
        print(f"  -> Characters: {orig_len:,} -> {new_len:,} (Removed {orig_len - new_len:,})")
        
    print("\nCleaning complete! Check the 'data/cleaned' folder.")

if __name__ == "__main__":
    main()
