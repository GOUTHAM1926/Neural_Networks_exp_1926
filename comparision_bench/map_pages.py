import re

with open('output.txt', 'r') as f:
    content = f.read()

pages = content.split('\x0c')
for i, page in enumerate(pages):
    page_num = i + 1
    # Clean up the page text to get the first few lines / main keywords
    lines = [line.strip() for line in page.split('\n') if line.strip()]
    if lines:
        header = " | ".join(lines[:3])
        print(f"Page {page_num}: {header[:100]}")
