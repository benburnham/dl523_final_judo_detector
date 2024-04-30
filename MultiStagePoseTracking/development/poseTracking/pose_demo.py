import cv2
import torch

# Open the input video
video_path = '../../FINAL DATASET/ALL/train/Uchi Mata/Uchi Mata_train_10.mp4'
input_video = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = input_video.get(cv2.CAP_PROP_FPS)
frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output_video = cv2.VideoWriter('output_video_pose.mp4', fourcc, fps, (frame_width, frame_height))

print(frame_width, frame_height, fps, frame_count)

# Process each frame
ret, frame = input_video.read()
# for _ in range(frame_count):
while ret:
    
    # Write the frame to the output video
    # cv2.imwrite("boxed_frame.jpg", frame)

    output_video.write(frame)
    ret, frame = input_video.read()

# Release the VideoCapture and VideoWriter objects
output_video.release()
input_video.release()
