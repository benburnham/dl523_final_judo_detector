num_of_frames = 300
batch_size = 1
num_workers = 0







import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.io import read_video
from torch.utils.data import DataLoader
import av

class VideoDataset(Dataset):
    def __init__(self, root_dir, mode='train', include_mirror=True, include_audio=False, num_frames=400):
        self.root_dir = root_dir
        self.mode = mode
        self.include_mirror = include_mirror
        self.include_audio = include_audio
        self.num_frames = num_frames

        if include_mirror:
            self.sub_dirs = ['ALL', 'NO_MIRROR']
        else:
            self.sub_dirs = ['ALL']

        self.data = []
        self.labels = []
        self.label_to_idx = {}
        current_label_id = 0

        for sub_dir in self.sub_dirs:
            sub_dir_path = os.path.join(self.root_dir, sub_dir, mode)
            techniques = os.listdir(sub_dir_path)
            for technique in techniques:
                technique_path = os.path.join(sub_dir_path, technique)
                videos = os.listdir(technique_path)
                for video in videos:
                    video_path = os.path.join(technique_path, video)
                    self.data.append(video_path)
                    if technique not in self.label_to_idx:
                        self.label_to_idx[technique] = current_label_id
                        current_label_id += 1
                    self.labels.append(self.label_to_idx[technique])

        

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data[idx]
        label_id = self.labels[idx]  # Use label ID instead of label string
        video, _, _ = read_video(video_path, pts_unit='sec')  # Load video

        # Permute and sample frames
        video = video.permute(0, 3, 1, 2)
        total_frames = video.shape[0]
        if total_frames > self.num_frames:
            frame_indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long()
            video = video[frame_indices]
        elif total_frames < self.num_frames:
            repeat_n = self.num_frames // total_frames + 1
            video = video.repeat(repeat_n, 1, 1, 1)[:self.num_frames]

        frames = [self.transform(frame) for frame in video]
        video_tensor = torch.stack(frames)

        return video_tensor, label_id

# Assuming necessary imports and initialization here
# Example usage:
# dataset = VideoDataset(root_dir='path/to/data', mode='train')
# loader = DataLoader(dataset, batch_size=10, shuffle=True)



root_dir = 'FINAL DATASET/'
#root_dir = 'FINAL_DATASET/FINAL_DATASET/FINAL_DATASET/'
#root_dir = '/projectnb/cs585/bhuvand/FINAL_DATASET/FINAL_DATASET/FINAL_DATASET/ALL'
#root_dir = '/content/drive/MyDrive/FINAL DATASET/ALL'

train_dataset = VideoDataset(root_dir=root_dir, mode='train', num_frames =num_of_frames)
eval_dataset = VideoDataset(root_dir=root_dir, mode='evaluate', num_frames =num_of_frames)






train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)



from torch import nn
from torchvision import models
from tqdm import tqdm

class Resnt18Rnn(nn.Module):
    def __init__(self, params_model):
        super(Resnt18Rnn, self).__init__()
        num_classes = params_model["num_classes"]
        dr_rate= params_model["dr_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]

        baseModel = models.resnet18(pretrained=pretrained)
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)
    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        ii = 0
        y = self.baseModel((x[:,ii]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, ts):
            y = self.baseModel((x[:,ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:,-1])
        out = self.fc1(out)
        return out

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

params_model = {
    "num_classes": 3,
    "dr_rate": 0.5,
    "pretrained": True,
    "rnn_hidden_size": 256,
    "rnn_num_layers": 2
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

model = Resnt18Rnn(params_model).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0000001)


num_epochs = 10

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    batch =0
    
    pbar = tqdm(total=len(data_loader), desc='Videos', leave=True)

    for videos, labels in data_loader:
        #print('batch number: ',batch)
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        running_loss += loss.item() * videos.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    
        
        pbar.update(1)
        pbar.set_postfix({'output': outputs})

    
    pbar.close()
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct / total
    print(epoch_loss)
    torch.cuda.empty_cache()
    return epoch_loss, epoch_acc

def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    
    
    pbar_eval = tqdm(total=len(data_loader), desc='Videos', leave=True)

    with torch.no_grad():
        for videos, labels in data_loader:
            #print('batch number: ',batch)
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print(outputs)
            
            pbar_eval.update(1)
            pbar_eval.set_postfix({'output': outputs})
            
            

    pbar_eval.close()
    eval_loss = running_loss / len(data_loader.dataset)
    eval_acc = correct / total
    torch.cuda.empty_cache()
    return eval_loss, eval_acc


train_a_l_save = []
eval_a_l_save = []

for epoch in range(num_epochs):
    print(f'epoch: {epoch}')
    print('Training')
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    train_a_l_save.append((train_loss, train_acc))
    print('Eval')
    eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device)
    eval_a_l_save.append((eval_loss, eval_acc))
    epoch_ = epoch
    
    checkpoint = {
    'epoch': epoch_ + 1,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()}
    file_name = 'model_'+ str(epoch) +'.pth'
    torch.save(checkpoint, file_name)
    


    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')
    print(f'Evaluation Loss: {eval_loss:.4f}, Evaluation Accuracy: {eval_acc:.4f}')
