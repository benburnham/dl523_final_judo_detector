import cv2
import numpy as np
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image

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

# Define the bounding box (x, y, width, height)
print(results["boxes"][0].detach().numpy())
print(int(results["boxes"][0].detach().numpy()))
x, y, w, h = int(results["boxes"][0].detach().numpy())
# Load the image
image = cv2.imread('example.jpg')

# Create a mask
mask = np.zeros_like(image[:, :, 0])
mask[y:y+h, x:x+w] = 255

# Apply mask to the image
person_masked = cv2.bitwise_and(image, image, mask=mask)


# Binarize the image (convert to binary)
ret, binary_image = cv2.threshold(person_masked, 127, 255, cv2.THRESH_BINARY)

# Invert the binary image
binary_image = cv2.bitwise_not(binary_image)

# Apply skeletonization
size = np.size(binary_image)
skel = np.zeros(binary_image.shape, np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
done = False

while not done:
    eroded = cv2.erode(binary_image, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(binary_image, temp)
    skel = cv2.bitwise_or(skel, temp)
    binary_image = eroded.copy()

    zeros = size - cv2.countNonZero(binary_image)
    if zeros == size:
        done = True

# Save the skeletonized image
cv2.imwrite("skeletonized_image.jpg", skel)
