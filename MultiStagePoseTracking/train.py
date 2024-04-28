'''
Ben Burnham
Demetrios Kechris

April 22, 2024
Deep Learning EC523
Judo technique classification using pose tracking
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import VideoDataset
from model import JudoTechniqueClassifier

# Training Loop
def train(model, train_loader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    # i = 0
    for i, (videos, labels) in enumerate(train_loader):
        print(videos[0])
        print('Label: ', labels[0])

        classID = model.class_to_classID(labels[0])
        # print(classID)
        
        # videos, labels = videos.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(videos[0])
        print('Prediction: ', model.classID_to_class(output))

        loss = criterion(output, classID)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Calculate percentage complete
        percent_complete = (i + 1) / len(train_loader) * 100
        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, {percent_complete:.2f}% Complete", end="\r")
    return running_loss / len(train_loader)

# Testing Loop
def test(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    i = 0
    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)

            # Calculate percentage complete
        percent_complete = (i + 1) / len(test_loader) * 100
        print(f"Testing: Batch [{i + 1}/{len(test_loader)}], Loss: {loss.item():.4f}, {percent_complete:.2f}% Complete", end="\r")
    accuracy = correct / total
    return accuracy


# Define data loaders
# data_dir = 'FINAL DATASET/'
data_dir = '../../FINAL DATASET/'
train_dataset = VideoDataset(data_dir, mode='train', include_mirror=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

test_dataset = VideoDataset(data_dir, mode='evaluate', include_mirror=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

# Print DataLoader stats
possible_techniques = train_dataset.get_techniques()
print("\nPossible techniques:", possible_techniques)
print("\nDataLoader Train Stats:")
print("Number of batches:", len(train_loader))
print("Number of samples:", len(train_loader.dataset))

print("\nDataLoader Test Stats:")
print("Number of batches:", len(test_loader))
print("Number of samples:", len(test_loader.dataset))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = JudoTechniqueClassifier(
    hidden_dim=256,
    layer_dim=3,
    num_outputs=3   # 'Osoto Gari'  'Seoi Nage'  'Uchi Mata'
    )
model.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
print("\n======================== Training started =========================")
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")
print("\n======================== Training finished ========================\n\n")

# Testing loop
test_accuracy = test(model, test_loader, criterion, device)
print(f"\nTest Accuracy: {test_accuracy:.2f}")

best_model_path = '../../'

torch.save(model.state_dict(), best_model_path)
