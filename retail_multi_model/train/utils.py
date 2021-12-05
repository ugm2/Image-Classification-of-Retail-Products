import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from retail_multi_model.train.dataset import FreiburgGroceriesDataset
np.random.seed(42)

logging.basicConfig(level=os.getenv("LOGGER_LEVEL", logging.WARNING))
logger = logging.getLogger(__name__)

def load_images_with_labels_from_folder(folder, num_images=None):
    images = []
    labels = []
    # Loop through each folder inside the main folder
    for folder_name in tqdm(os.listdir(folder), desc='Loading images'):
        # Check if the folder is a directory
        if os.path.isdir(os.path.join(folder, folder_name)):
            # Loop through each image inside the folder
            for image_name in os.listdir(os.path.join(folder, folder_name)):
                # Check if the image is a file and ends with .png
                if os.path.isfile(os.path.join(folder, folder_name, image_name)) \
                    and image_name.endswith(".png"):
                    # Load the image and the label
                    image = Image.open((os.path.join(folder, folder_name, image_name)))
                    label = folder_name
                    # Add the image and the label to the list
                    images.append(image.copy())
                    labels.append(label)
                    image.close()
                    # Check if the number of images is reached
                    if num_images is not None and len(images) >= num_images:
                        break
    # Log length
    logger.info("Loaded %d images", len(images))
    return images, labels
    

def prepare_dataset(images,
                    labels,
                    model,
                    test_size=.2,
                    train_transform=None,
                    val_transform=None):
    logger.info("Preparing dataset")
    # Split the dataset in train and test
    images_train, images_test, labels_train, labels_test = \
        train_test_split(images, labels, test_size=test_size)

    # Preprocess images using model feature extractor
    images_train = model.preprocess_image(images_train)
    images_test = model.preprocess_image(images_test)

    # Create the datasets
    train_dataset = FreiburgGroceriesDataset(images_train, labels_train, train_transform)
    test_dataset = FreiburgGroceriesDataset(images_test, labels_test, val_transform)
    logger.info("Train dataset: %d images", len(labels_train))
    logger.info("Test dataset: %d images", len(labels_test))
    return train_dataset, test_dataset