import cv2
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
import numpy as np
from box_mse import find_closest_box

def print_results(results):
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        if label.item() == 1:
            print(f"Detected person with confidence "    
                f"{round(score.item(), 3)} at location {box}")

            int_array = [int(x) for x in box]
            x1, y1, x2, y2 = int_array
            frame = cv2.rectangle(frame, (x1,y1), (x2,y2), color=(0, 255, 0), thickness=2)
    return


def video_to_tensor(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to tensor and append to list
        frame_tensor = np.array(frame)
        frames.append(frame_tensor)
    cap.release()
    return np.stack(frames) # Convert to tensor

# Example usage
video_path = "../../FINAL DATASET/ALL/train/Uchi Mata/Uchi Mata_train_10.mp4"
video_tensor = video_to_tensor(video_path)
print("Shape of video tensor:", video_tensor.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr").to(device)

inputs = processor(images=video_tensor, return_tensors="pt").to(device)
print("Shape of input:", inputs['pixel_values'].shape)
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.7
target_sizes = torch.tensor(video_tensor.size)
results =  processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

print_results(results)

boxes = results["boxes"]
boxes = [int(i) for i in boxes.tolist()]
find_closest_box(boxes, 1280, 720)