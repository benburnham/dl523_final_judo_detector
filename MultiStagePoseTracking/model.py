import torch
import torch.nn as nn
import numpy as np
from mmpose.apis import MMPoseInferencer

class JudoTechniqueClassifier(torch.nn.Module):
    def __init__(self, hidden_dim, layer_dim, dropout_rate, num_outputs, device):
        super(JudoTechniqueClassifier, self).__init__()
        self.num_outputs = num_outputs
        self.device = device

        # Pose Detection: 
        # det_model='mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py'
        # det_weights='../../rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth'
        det_model='mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py',
        det_weights='../../rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth',
        self.pose_detection_model = MMPoseInferencer(
            pose2d='rtmo',
            det_model=det_model, 
            det_weights=det_weights,
            det_cat_ids=[0],  # the category id of 'human' class
            device=device
        )

        # LSTM Claasification
        input_dim = 2   # 2 pose sequences, 1 for each combatant
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, 
                            hidden_dim, 
                            layer_dim, 
                            dropout=dropout_rate,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, num_outputs)

        # Softmax activation for classification function
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
                pose1_seq.append(torch.tensor(frame_detections[0]['keypoints']).to(self.device))
                pose2_seq.append(torch.tensor(frame_detections[1]['keypoints']).to(self.device))

            # If only one pose detected, assign to both sequences
            elif len(frame_detections) == 1:
                pose1_seq.append(torch.tensor(frame_detections[0]['keypoints']).to(self.device))
                pose2_seq.append(torch.tensor(frame_detections[0]['keypoints']).to(self.device))

            # If none are detected, skip
            else:
                pass

        lstm_input = self.prepare_lstm_input(pose1_seq, pose2_seq)
        # print(lstm_input.shape)

        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.layer_dim, lstm_input.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.layer_dim, lstm_input.size(0), self.hidden_dim).to(self.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        lstm_out = self.dropout(lstm_out)       # Add dropout
        lstm_out = torch.mean(lstm_out, dim=(0, 1), keepdim=True)
        predictions = self.fc(lstm_out)    # Linear layer
        
        return predictions
    
    def prepare_lstm_input(self, pose1_seq, pose2_seq):

        # Convert lists to tensors
        pose1_seq = torch.stack(pose1_seq)
        pose2_seq = torch.stack(pose2_seq)
        
        # Prepare data for LSTM classification
        pose_seqs = torch.stack([pose1_seq, pose2_seq], dim=1)
        lstm_input = pose_seqs.view(pose_seqs.size(0), -1, 2)
        
        return lstm_input

    def class_to_classID(self, technique):
        # 'Osoto Gari'  'Seoi Nage'  'Uchi Mata'
        one_hot = torch.zeros(self.num_outputs).to(self.device)
        if technique == 'Osoto Gari':
            one_hot[0] = 1
        elif technique == 'Seoi Nage':
            one_hot[1] = 1
        elif technique == 'Uchi Mata':
            one_hot[2] = 1

        return one_hot.unsqueeze(0)
    
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

        
        # # Look at detections in each frame
        # for frame in detections:
        #     frame_detections = frame['predictions'][0]

        #     # If more than one pose detected, take two largest poses
        #     if len(frame_detections) > 1:
        #         frame_detections.sort(key=lambda x: (x['bbox'][0][2] - x['bbox'][0][0]) * (x['bbox'][0][3] - x['bbox'][0][1]), reverse=True)
        #         pose1_seq.append(frame_detections[0]['keypoints'])
        #         pose2_seq.append(frame_detections[1]['keypoints'])
            
        #     # If only one pose detected, assign to both sequences
        #     elif len(frame_detections) == 1:
        #         pose1_seq.append(frame_detections[0]['keypoints'])
        #         pose2_seq.append(frame_detections[0]['keypoints'])
            
        #     # If none are detected, skip
        #     else:
        #         pass
    
    # def person_detections(self, frame):
    #     inputs = self.processor(images=frame, return_tensors="pt")
    #     # print("Shape of input:", inputs['pixel_values'].shape)
    #     outputs = self.person_detection_model(**inputs)

    #     # Convert outputs and keep detections with score > 0.7
    #     target_sizes = torch.tensor(frame.size)
    #     results =  self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

    #     boxes = results["boxes"]
    #     boxes = [int(i) for i in boxes.tolist()]

    #     return self.find_closest_box(boxes, 1280, 720)

    # def find_closest_box(self, boxes, frame_width, frame_height):
    #     center_x = frame_width // 2
    #     center_y = frame_height // 2

    #     box_centers_x = (boxes[:, 0, 0] + boxes[:, 1, 0]) // 2
    #     box_centers_y = (boxes[:, 0, 1] + boxes[:, 1, 1]) // 2

    #     distances = np.sqrt((box_centers_x - center_x)**2 + (box_centers_y - center_y)**2)
    #     closest_indices = np.argsort(distances)

    #     closest_box1 = boxes[closest_indices[0]]
    #     closest_box2 = boxes[closest_indices[1]] if len(boxes) > 1 else None

    #     return closest_box1, closest_box2 
