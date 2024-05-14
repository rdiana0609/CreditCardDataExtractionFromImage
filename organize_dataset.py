import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path


def create_directory_structure(base_path):
    for split in ['train', 'test']:
        for digit in range(10):
            Path(f"{base_path}/{split}/{digit}").mkdir(parents=True, exist_ok=True)


def organize_dataset(src_path, dest_path, test_size=0.2):
    # Create the directory structure
    create_directory_structure(dest_path)

    # Iterate over each digit directory
    for digit in range(10):
        digit_path = Path(src_path) / str(digit)
        images = list(digit_path.glob("*.png"))

        # Split images into training and testing sets
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)

        # Move training images
        for img in train_images:
            shutil.copy(img, f"{dest_path}/train/{digit}/{img.name}")

        # Move testing images
        for img in test_images:
            shutil.copy(img, f"{dest_path}/test/{digit}/{img.name}")


if __name__ == "__main__":
    # Path to the source dataset directory
    src_path = "dataset"

    # Path to the destination directory
    dest_path = "split_dataset"

    # Organize the dataset with 80% training and 20% testing
    organize_dataset(src_path, dest_path, test_size=0.2)
