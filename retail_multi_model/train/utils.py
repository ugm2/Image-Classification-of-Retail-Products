import os
import numpy as np
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from retail_multi_model.train.dataset import RetailDataset
from transformers import BatchFeature
from PIL import Image
np.random.seed(42)

logging.basicConfig(level=os.getenv("LOGGER_LEVEL", logging.WARNING))
logger = logging.getLogger(__name__)
    
def prepare_dataset(images,
                    labels,
                    model,
                    test_size=.2,
                    train_transform=None,
                    val_transform=None,
                    batch_size=512):
    logger.info("Preparing dataset")
    # Split the dataset in train and test
    images_train, images_test, labels_train, labels_test = \
        train_test_split(images, labels, test_size=test_size)

    # Preprocess images using model feature extractor
    images_train_prep = []
    images_test_prep = []
    for bs in tqdm(range(0, len(images_train), batch_size), desc="Preprocessing training images"):
        images_train_batch = [Image.fromarray(np.array(image)) for image in images_train[bs:bs+batch_size]]
        images_train_batch = model.preprocess_image(images_train_batch)
        images_train_prep.append(images_train_batch['pixel_values'])
    for bs in tqdm(range(0, len(images_test), batch_size), desc="Preprocessing test images"):
        images_test_batch = [Image.fromarray(np.array(image)) for image in images_test[bs:bs+batch_size]]
        images_test_batch = model.preprocess_image(images_test_batch)
        images_test_prep.append(images_test_batch['pixel_values'])

    # Flatten the lists
    images_train_prep = [item for sublist in images_train_prep for item in sublist]
    images_test_prep = [item for sublist in images_test_prep for item in sublist]

    # Create BatchFeatures
    images_train_prep = {"pixel_values": images_train_prep}
    train_batch_features = BatchFeature(data=images_train_prep)
    images_test_prep = {"pixel_values": images_test_prep}
    test_batch_features = BatchFeature(data=images_test_prep)

    # Create the datasets
    train_dataset = RetailDataset(train_batch_features, labels_train, train_transform)
    test_dataset = RetailDataset(test_batch_features, labels_test, val_transform)
    logger.info("Train dataset: %d images", len(labels_train))
    logger.info("Test dataset: %d images", len(labels_test))
    return train_dataset, test_dataset

