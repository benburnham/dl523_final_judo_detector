from torch.utils.data import DataLoader
from torchvision.io import read_video
from dataloader import VideoDataset

data_dir = '../../FINAL DATASET/'
train_dataset = VideoDataset(data_dir, mode='train', include_mirror=True, include_audio=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

evaluate_dataset = VideoDataset(data_dir, mode='evaluate', include_mirror=True, include_audio=False)
evaluate_loader = DataLoader(evaluate_dataset, batch_size=32, shuffle=True)

# Print DataLoader stats
print("DataLoader Train Stats:")
print("Number of batches:", len(train_loader))
print("Number of samples:", len(train_loader.dataset))

print("\nDataLoader Evaluate Stats:")
print("Number of batches:", len(evaluate_loader))
print("Number of samples:", len(evaluate_loader.dataset))