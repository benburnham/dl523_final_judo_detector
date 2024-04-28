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
print("Number of training batches:", len(train_loader))
print("Number of testing batches:", len(test_loader))
# print("Number of samples:", len(train_loader.dataset))
# print("Number of samples:", len(test_loader.dataset))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize model
model = JudoTechniqueClassifier(
    hidden_dim=256,
    layer_dim=3,
    dropout_rate=0.5,
    num_outputs=3,  # 'Osoto Gari'  'Seoi Nage'  'Uchi Mata'
    device=device
    )
model.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
print("\n======================== Training started =========================")
num_epochs = 10
verbose=False

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (videos, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(videos[0])

        classID = model.class_to_classID(labels[0])
        loss = criterion(output.squeeze(0), classID.type_as(output))
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        # Calculate percentage complete
        percent_complete = (i + 1) / len(train_loader) * 100
        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, {percent_complete:.2f}% Complete")
        
        if verbose:
            print('Video: ', videos[0])
            print('Label: ', labels[0])
            print('Prediction: ', model.classID_to_class(output))
            print(loss)

    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} complete, Train Loss: {train_loss:.4f}\n")

print("\n======================== Training finished ========================\n")
print("\n========================== Begin Testing ==========================\n")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i, (videos, labels) in enumerate(test_loader):
        output = model(videos[0])
        classID = model.class_to_classID(labels[0])
        # loss = criterion(output.squeeze(0), classID.type_as(output))

        total += 1
        prediction = model.classID_to_class(output)
        if prediction == labels[0]:
            correct +=1

        # Calculate percentage complete
        percent_complete = (i + 1) / len(test_loader) * 100
        print(f"Testing [{i + 1}/{len(test_loader)}], Label: {labels[0]}, Prediction: {prediction}, {percent_complete:.2f}% Complete")

test_accuracy = correct / total
print(f"\nTest Accuracy: {test_accuracy:.2f}")

best_model_path = '../../'
torch.save(model.state_dict(), best_model_path)