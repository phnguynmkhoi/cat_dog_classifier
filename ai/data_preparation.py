import os

from sklearn.model_selection import train_test_split
import shutil

# create training and validation directories
def create_data_directories(train_dir, TEST_DIR):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

# create a function to split the dataset into training and validation sets
def split_dataset(base_dir, train_dir, TEST_DIR, test_size=0.2, label="null"):
    # Get all image files in the base directory
    all_images = [f for f in os.listdir(base_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    # Split the dataset into training and validation sets
    train_images, val_images = train_test_split(all_images, test_size=test_size)
    
    # Copy images to the respective directories
    for image in train_images:
        shutil.copy(os.path.join(base_dir, image), os.path.join(train_dir, f"{label}_{image}"))
    
    for image in val_images:
        shutil.copy(os.path.join(base_dir, image), os.path.join(TEST_DIR, f"{label}_{image}"))

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

DOG_IMAGE_DIR = 'data/PetImages/Dog'
CAT_IMAGE_DIR = 'data/PetImages/Cat'

# Create directories for training and validation sets
create_data_directories(TRAIN_DIR, TEST_DIR)
# Split the dataset into training and validation sets
split_dataset(DOG_IMAGE_DIR, TRAIN_DIR, TEST_DIR, label="dog")
split_dataset(CAT_IMAGE_DIR, TRAIN_DIR, TEST_DIR, label="cat")
