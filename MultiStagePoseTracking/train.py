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
from tqdm import tqdm


# Define data loaders
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize model
model = JudoTechniqueClassifier(hidden_dim=256,
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

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (videos, labels) in progress_bar:
        optimizer.zero_grad()
        output = model(videos[0])
        if output is None:
            tqdm.write('Bad clip: {}'.format(videos[0]))
            progress_bar.set_description(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}]')
            continue

        classID = model.class_to_classID(labels[0])
        loss = criterion(output.squeeze(0), classID.type_as(output))
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        if verbose:
            tqdm.write('Video: {}'.format(videos[0]))
            tqdm.write('Label: {}'.format(labels[0]))
            tqdm.write('Prediction: {}'.format(model.classID_to_class(output)))
            tqdm.write('Loss: {}'.format(loss.item()))

        # Update tqdm progress bar description
        progress_bar.set_description(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    train_loss = running_loss / len(train_loader)
    tqdm.write(f"Epoch {epoch+1} complete, Train Loss: {train_loss:.4f}")

print("\n======================== Training finished ========================\n")
print("\n========================== Begin Testing ==========================\n")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, (videos, labels) in progress_bar:
        output = model(videos[0])
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

best_model_path = 'trained_judo_classifier2.pth'
torch.save(model.state_dict(), best_model_path)