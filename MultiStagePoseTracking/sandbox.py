from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image
import requests
import cv2

# Load the video
video_path = '../../FINAL DATASET/ALL/train/Uchi Mata/Uchi Mata_train_20.mp4'
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = cap.read()

# Check if the frame was read successfully
if not ret:
    print("Error: Failed to read the first frame from the video.")
    exit()

# Convert the frame from BGR to RGB format
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Create a PIL Image object from the frame
pil_image = Image.fromarray(frame_rgb)

# Save the PIL Image as example.jpg
output_path = 'example.jpg'
pil_image.save(output_path)

# Release the video capture object
cap.release()

print("Image saved as example.jpg")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")

inputs = processor(images=pil_image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.7
target_sizes = torch.tensor([pil_image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {model.config.id2label[label.item()]} with confidence "
          f"{round(score.item(), 3)} at location {box}")
