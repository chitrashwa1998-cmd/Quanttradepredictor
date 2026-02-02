import os
import re

FOLDERS = ["utils", "pages"]
TARGET_NUMBERS = {"52168", "49543", "37054", "49229"}
REPLACEMENT = "59182"

# Create regex like: \b(52168|49543|37054)\b
pattern = re.compile(r"\b(" + "|".join(TARGET_NUMBERS) + r")\b")

def replace_in_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_content, count = pattern.subn(REPLACEMENT, content)

    if count > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated {count} occurrence(s) in {file_path}")

for folder in FOLDERS:
    if not os.path.isdir(folder):
        print(f"Warning: folder '{folder}' not found")
        continue

    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".py"):
                replace_in_file(os.path.join(root, file))
