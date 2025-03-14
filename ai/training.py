import os
import tqdm

import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from data_loader import ImageDataset, transform

# create datasets and dataloaders
train_dataset = ImageDataset('data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# create ResNet model
model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming binary classification (dog/cat)
# define loss function and optimizer
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model
num_epochs = 2
for epoch in tqdm.tqdm(range(num_epochs)):
    print(f"Epoch: {epoch+1}/{num_epochs}")
    # Save the model every 2 epochs
    if epoch > 0:
        torch.save(model.state_dict(), f'model/model_epoch_{epoch}.pth')
    model.train()
    running_loss = 0.0
    for images, labels in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        # Convert labels to tensor
        labels = torch.tensor([0 if label == 'dog' else 1 for label in labels])
        loss_value = loss(outputs, labels)
        loss_value.backward()
        optimizer.step()
        running_loss += loss_value.item()
        print(f"Batch Loss: {loss_value.item():.4f} for epoch {epoch+1}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

torch.save(model, 'model/model_final.pth')