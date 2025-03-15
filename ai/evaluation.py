import tqdm
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loader import ImageDataset, transform
from torchvision.models import resnet18
import torch.nn as nn

# create transformations for the images

val_dataset = ImageDataset('data/test', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

model = torch.load("model/model_final.pth", weights_only=False)
model.eval()  # Set to evaluation mode

# Initialize lists to store true labels and predictions
y_true = []
y_pred = []
# Iterate through the validation set
with torch.no_grad():
    for images, labels in tqdm.tqdm(val_loader):
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # Convert labels to tensor
        labels = torch.tensor([0 if label == 'dog' else 1 for label in labels])
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
# Print metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
