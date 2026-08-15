from pathlib import Path

EXTRACTED_FOLDER = Path("data/extracted")

txt_files = list(EXTRACTED_FOLDER.glob("*.txt"))

print(f"Found {len(txt_files)} extracted files.\n")

for file in txt_files:
    text = file.read_text(encoding="utf-8", errors="replace")

    print("=" * 70)
    print(f"FILE: {file.name}")
    print(f"Characters: {len(text):,}")
    print(f"Lines: {len(text.splitlines()):,}")
    print("\nFIRST 1500 CHARACTERS:\n")
    print(text[:1500])
    print()
