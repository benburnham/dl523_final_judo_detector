'''
JUDO Throw Identifier
Ben Burnham
April 21, 2024
Deep Learning EC523
'''

import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from km_model import KMJudoTechniqueClassifier

class VideoDataset(Dataset):
    def __init__(self, root_dir, mode='train', include_mirror=True):
        self.root_dir = root_dir
        self.mode = mode
        self.include_mirror = include_mirror

        if include_mirror:
            self.sub_dirs = ['ALL']
        else:
            self.sub_dirs = ['NO_MIRROR']

        self.data = []
        for sub_dir in self.sub_dirs:
            sub_dir_path = os.path.join(self.root_dir, sub_dir, mode)
            techniques = os.listdir(sub_dir_path)
            for technique in techniques:
                technique_path = os.path.join(sub_dir_path, technique)
                poses = [os.path.join(technique_path, pose) for pose in os.listdir(technique_path)]
                self.data.extend(poses)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pose_path = self.data[idx]
        label = os.path.basename(os.path.dirname(pose_path))
        return pose_path, label
        
    def get_techniques(self):
        techniques = set()
        for video_path in self.data:
            technique = os.path.basename(os.path.dirname(video_path))
            techniques.add(technique)
        return list(techniques)
    
if __name__ == "__main__":
    # Define data loaders
    data_dir = 'Pose_Dataset'
    train_dataset = VideoDataset(data_dir, mode='train', include_mirror=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

    test_dataset = VideoDataset(data_dir, mode='evaluate', include_mirror=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    # Print DataLoader stats
    possible_techniques = train_dataset.get_techniques()
    print("\nPossible techniques:", possible_techniques)
    print("Number of training batches:", len(train_loader))
    print("Number of testing batches:", len(test_loader))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize model
    model = KMJudoTechniqueClassifier(hidden_dim=256,
                                    layer_dim=3,
                                    dropout_rate=0.5,
                                    num_outputs=3,  # 'Osoto Gari'  'Seoi Nage'  'Uchi Mata'
                                    device=device)
    model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\n======================== Training started =========================")
    num_epochs = 5
    verbose=False
    #best_model_path = 'pre_pose_KM_judo_classifier.pth'
    best_model_path = 'pre_pose_KM_judo_classifier_10e.pth'

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (pose_path, labels) in progress_bar:
            with open(pose_path[0], 'r') as json_file:
                pose_data = json.load(json_file)
            optimizer.zero_grad()
            output = model(pose_data)

            if output is None:
                tqdm.write('Bad clip: {}'.format(pose_path[0]))
                progress_bar.set_description(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}]')
                continue

            classID = model.class_to_classID(labels[0])
            loss = criterion(output.squeeze(0), classID.type_as(output))
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            # Update tqdm progress bar description
            progress_bar.set_description(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            if verbose:
                tqdm.write('Video: {}'.format(pose_path[0]))
                tqdm.write('Label: {}'.format(labels[0]))
                tqdm.write('Prediction: {}'.format(model.classID_to_class(output)))
                tqdm.write('Loss: {}'.format(loss.item()))

        train_loss = running_loss / len(train_loader)
        tqdm.write(f"Epoch {epoch+1} complete, Train Loss: {train_loss:.4f}")
        torch.save(model.state_dict(), best_model_path)

    print("\n======================== Training finished ========================\n")
    print("\n========================== Begin Testing ==========================\n")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, (pose_path, labels) in progress_bar:
            with open(pose_path[0], 'r') as json_file:
                pose_data = json.load(json_file)
            output = model(pose_data)
            classID = model.class_to_classID(labels[0])
            # loss = criterion(output.squeeze(0), classID.type_as(output))

            total += 1
            prediction = model.classID_to_class(output)
            if prediction == labels[0]:
                correct +=1

            # Update tqdm progress bar description
            progress_bar.set_description(f'Testing [{i + 1}/{len(test_loader)}], Label: {labels[0]}, Prediction: {prediction}')

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%")


