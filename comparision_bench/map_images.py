import subprocess
import os

# Get list of images from pdfimages
result = subprocess.run(['pdfimages', '-list', 'Rough Draft - Results.pdf'], capture_output=True, text=True)
lines = result.stdout.split('\n')[2:] # skip headers

image_to_page = {}
for line in lines:
    parts = line.split()
    if len(parts) > 2:
        page = int(parts[0])
        num = int(parts[1])
        image_to_page[f"ext_img-{num:03d}.png"] = page

# Get sizes of top images
files = [f for f in os.listdir('.') if f.startswith('ext_img-') and f.endswith('.png')]
sizes = [(f, os.path.getsize(f)) for f in files]
sizes.sort(key=lambda x: x[1], reverse=True)

for f, size in sizes[:20]:
    page = image_to_page.get(f, 'Unknown')
    print(f"{f}: {size/1024:.1f} KB -> Page {page}")

