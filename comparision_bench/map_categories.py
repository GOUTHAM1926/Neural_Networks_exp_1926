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

categories = {
    "Attention": (3, 12),
    "Loss": (27, 36),
    "Norm_Activations": (13, 26),
    "Memory": (84, 92),
    "DDP": (93, 96)
}

best_images = {}
for cat, (start, end) in categories.items():
    cat_images = []
    for img, page in image_to_page.items():
        if start <= page <= end:
            if os.path.exists(img):
                size = os.path.getsize(img)
                cat_images.append((img, size))
    
    # Sort by size and take the top 2
    cat_images.sort(key=lambda x: x[1], reverse=True)
    best_images[cat] = [img for img, size in cat_images[:2]]

print(best_images)
