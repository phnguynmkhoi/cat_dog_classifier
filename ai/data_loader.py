import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform

        self.image_labels = [(f"{directory}/{file_name}", file_name.split("_")[0]) for file_name in os.listdir(directory) if file_name.endswith('.jpg') or file_name.endswith('.png')]

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_path, label = self.image_labels[idx]
        image = Image.open(image_path).convert('RGB')  # Convert all images to RGB
        if self.transform:
            image = self.transform(image)

        return image, label
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])