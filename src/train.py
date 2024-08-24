from tqdm import tqdm
from src.config import device, batch_size
from src.datasets import train_loader, val_loader
from src.model import optimizer, model, criterion
import torch


def train(epochs = 5, save_model_to = 'data/model.pth'):
    for epoch in range(epochs):
        for (data, target) in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            data, target = data.float().to(device), target.float().to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item() / batch_size}')

        with torch.no_grad():
            total = 0
            correct = 0
            for (data, target) in tqdm(val_loader, desc=f'Validation for Epoch {epoch + 1}'):
                data, target = data.float().to(device), target.float().to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            print(f'Accuracy: {100 * correct / total}%')

    if save_model_to is not None:
        torch.save(model.state_dict(), save_model_to)



def load_model(model_path):
    model.load_state_dict(torch.load(model_path))
