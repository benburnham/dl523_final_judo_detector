import cv2
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image

def get_objects(image):
    # Convert the frame from BGR to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a PIL Image object from the frame
    pil_image = Image.fromarray(frame_rgb)
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
        if model.config.id2label[label.item()] == 'person':
            print(f"Detected {model.config.id2label[label.item()]} with confidence "    
                f"{round(score.item(), 3)} at location {box}")

            int_array = [int(x) for x in box]
            x1, y1, x2, y2 = int_array
            image = cv2.imread('example.jpg')
            cv2.rectangle(image, (x1,y1), (x2,y2), color=(0, 255, 0), thickness=2   )
            cv2.imwrite("boxed_example.jpg", image)

# Open the input video
video_path = '../../FINAL DATASET/ALL/train/Uchi Mata/Uchi Mata_train_10.mp4'
input_video = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = input_video.get(cv2.CAP_PROP_FPS)
frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

# Process each frame
for _ in range(frame_count):
    ret, frame = input_video.read()

    if not ret:
        break

    get_objects(frame)

    # Write the frame to the output video
    output_video.write(frame)

# Release the VideoCapture and VideoWriter objects
input_video.release()
output_video.release()
