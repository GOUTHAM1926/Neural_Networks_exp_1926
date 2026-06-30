import re

with open('results.txt', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "ms" in line or "GB/s" in line or "TFLOPS" in line:
        print(f"Line {i}: {line.strip()}")
