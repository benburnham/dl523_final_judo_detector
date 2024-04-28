import cv2
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image

# Open the input video
video_path = '../../FINAL DATASET/ALL/train/Uchi Mata/Uchi Mata_train_111.mp4'
input_video = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = input_video.get(cv2.CAP_PROP_FPS)
frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output_video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

print(frame_width, frame_height, fps, frame_count)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr").to(device)

# Process each frame
ret, frame = input_video.read()
# for _ in range(frame_count):
while ret:
    # Convert the frame from BGR to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a PIL Image object from the frame
    pil_image = Image.fromarray(frame_rgb)

    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.7
    target_sizes = torch.tensor([pil_image.size[::-1]])
    results =  processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        if label.item() == 1:
            print(f"Detected person with confidence "    
                f"{round(score.item(), 3)} at location {box}")

            int_array = [int(x) for x in box]
            x1, y1, x2, y2 = int_array
            frame = cv2.rectangle(frame, (x1,y1), (x2,y2), color=(0, 255, 0), thickness=2)

    # Write the frame to the output video
    # cv2.imwrite("boxed_frame.jpg", frame)

    output_video.write(frame)
    ret, frame = input_video.read()

# Release the VideoCapture and VideoWriter objects
output_video.release()
input_video.release()
