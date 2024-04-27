import torch
import torch.nn as nn
import numpy as np
from mmpose.apis import MMPoseInferencer

# import timm
# from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
# import cv2  # For video processing

class JudoTechniqueClassifier(torch.nn.Module):
    def __init__(self, hidden_dim, layer_dim, num_outputs):
        super(JudoTechniqueClassifier, self).__init__()
        self.num_outputs = num_outputs

        # # Person Detection: Deformable DETR model with ResNet-50 backbone
        # self.processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
        # self.person_detection_model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")

        # Pose Detection: 
        det_model='mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py'
        det_weights='../../rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth'
        self.pose_detection_model = MMPoseInferencer(
            pose2d='rtmo',
            det_model=det_model, 
            det_weights=det_weights,
            det_cat_ids=[0]  # the category id of 'human' class
        )

        # LSTM Claasification
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.Sequential(
            nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True),
            nn.Linear(hidden_dim, num_outputs)
        )
    
    def forward(self, video):
        # get poses
        detection_generator = self.pose_detection_model(video)
        detections = [result for result in detection_generator]


        # # Loop through video frames
        # for frame in video:
        #     # Perform person detection on the frame
        #     predictions = next()
        #     keypoints = [prediction['keypoints'] for prediction in predictions['predictions'][0]]
            
        #     # Perform pose tracking on the detected keypoints
        #     tracked_poses = self.pose_tracking_model.track(keypoints)
            
        #     # Extract features from the tracked poses
        #     pose_features = self.extract_features(tracked_poses)
            
        #     # Store the poses and features
        #     # poses.append(tracked_poses)
        #     features.append(pose_features)


        
        # Prepare data for LSTM classification
        lstm_input = self.prepare_lstm_input(features)
        
        # Perform LSTM classification
        predictions = self.lstm_model(lstm_input)
        
        # Convert predictions to one-hot encoding
        one_hot = torch.zeros(self.num_outputs)
        max_idx = torch.argmax(predictions)
        one_hot[max_idx] = 1
        
        return one_hot
    
    def extract_features(self, tracked_poses):
        # Implement feature extraction from tracked poses
        # This can include calculating joint angles, velocities, etc.
        # Return a feature vector for each frame
        pass

    def prepare_lstm_input(self, features):
        # Implement data preparation for LSTM classification
        # This may involve padding sequences, converting to tensor, etc.
        # Return the prepared data
        pass

    def person_detections(self, frame):
        inputs = self.processor(images=frame, return_tensors="pt")
        # print("Shape of input:", inputs['pixel_values'].shape)
        outputs = self.person_detection_model(**inputs)

        # Convert outputs and keep detections with score > 0.7
        target_sizes = torch.tensor(frame.size)
        results =  self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

        boxes = results["boxes"]
        boxes = [int(i) for i in boxes.tolist()]

        return self.find_closest_box(boxes, 1280, 720)

    def find_closest_box(self, boxes, frame_width, frame_height):
        center_x = frame_width // 2
        center_y = frame_height // 2

        box_centers_x = (boxes[:, 0, 0] + boxes[:, 1, 0]) // 2
        box_centers_y = (boxes[:, 0, 1] + boxes[:, 1, 1]) // 2

        distances = np.sqrt((box_centers_x - center_x)**2 + (box_centers_y - center_y)**2)
        closest_indices = np.argsort(distances)

        closest_box1 = boxes[closest_indices[0]]
        closest_box2 = boxes[closest_indices[1]] if len(boxes) > 1 else None

        return closest_box1, closest_box2 
