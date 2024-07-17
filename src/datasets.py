from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from src.transform import ToNumpy, ExtractKeypoints, hands, HandleGestureDataset
from src.config import batch_size

train_transform = transforms.Compose([
    transforms.RandomRotation((-90, 90)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    ToNumpy(),
    ExtractKeypoints(hands),
    HandleGestureDataset(),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    ExtractKeypoints(hands),
    HandleGestureDataset(),
    transforms.ToTensor(),
])

dataset = ImageFolder('data/rps', transform=train_transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
