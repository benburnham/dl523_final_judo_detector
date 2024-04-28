import torch
import torch.nn as nn
import numpy as np
from mmpose.apis import MMPoseInferencer
from sklearn.preprocessing import MinMaxScaler

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
        # det_model='mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py'
        # det_weights='../../rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth'
        det_model='mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py',
        det_weights='../../rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth',
        self.pose_detection_model = MMPoseInferencer(
            pose2d='rtmo',
            det_model=det_model, 
            det_weights=det_weights,
            det_cat_ids=[0]  # the category id of 'human' class
        )

        # LSTM Claasification
        # Input: 2 pose sequences
        input_dim = 2  
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm_model = nn.Sequential(
            nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True),
            nn.Linear(hidden_dim, num_outputs)
        )

        # Softmax activation for classification
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, video):
        # Get poses
        detection_generator = self.pose_detection_model(video)
        detections = [result for result in detection_generator]
        
        pose1_seq = []
        pose2_seq = []

        # Look at detections in each frame
        for frame in detections:
            frame_detections = frame['predictions'][0]

            # If more than one pose detected, take two largest poses
            if len(frame_detections) > 1:
                frame_detections.sort(key=lambda x: (x['bbox'][0][2] - x['bbox'][0][0]) * (x['bbox'][0][3] - x['bbox'][0][1]), reverse=True)
                pose1_seq.append(frame_detections[0]['keypoints'])
                pose2_seq.append(frame_detections[1]['keypoints'])
            
            # If only one pose detected, assign to both sequences
            elif len(frame_detections) == 1:
                pose1_seq.append(frame_detections[0]['keypoints'])
                pose2_seq.append(frame_detections[0]['keypoints'])
            
            # If none are detected, skip
            else:
                pass
        
        # Prepare data for LSTM classification
        lstm_input = self.prepare_lstm_input(pose1_seq, pose2_seq)
        
        # Perform LSTM classification
        predictions = self.lstm_model(lstm_input)
        
        return predictions
    
    def prepare_lstm_input(X_combatant1, X_combatant2):
        # Normalize sequences
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_combatant1_scaled = scaler.fit_transform(X_combatant1.reshape(-1, X_combatant1.shape[-1])).reshape(X_combatant1.shape)
        X_combatant2_scaled = scaler.transform(X_combatant2.reshape(-1, X_combatant2.shape[-1])).reshape(X_combatant2.shape)
        
        return X_combatant1_scaled, X_combatant2_scaled

    def class_to_classID(self, technique):
        # 'Osoto Gari'  'Seoi Nage'  'Uchi Mata'
        one_hot = torch.zeros(self.num_outputs)
        if technique == 'Osoto Gari':
            one_hot[0] = 1
        elif technique == 'Seoi Nage':
            one_hot[1] = 1
        elif technique == 'Uchi Mata':
            one_hot[2] = 1

        return one_hot
    
    def classID_to_class(self, predictions):
        # class dictionary
        class_mapping_reverse = {
            0: 'Osoto Gari',
            1: 'Seoi Nage',
            2: 'Uchi Mata'
        }

        # get highest prediction and get technique name
        predictions = self.softmax(predictions)
        classID = torch.argmax(predictions).item()
        return class_mapping_reverse[classID]

        
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
