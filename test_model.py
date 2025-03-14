# ...existing code...
import torch
from torchvision.models import resnet18
import torch.nn as nn

# Define the model architecture first
model = resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Same architecture as in training

# Then load the state dictionary
model.load_state_dict(torch.load('model/model_final.pth', map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode

# ...existing code...

torch.save(model, 'model/model_final.pth')