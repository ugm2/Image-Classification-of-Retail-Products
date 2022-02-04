import io
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from retail_multi_model.train.dataset import FreiburgGroceriesDataset
import hashlib
import time
from imagines import DatasetAugmentation
np.random.seed(42)

logging.basicConfig(level=os.getenv("LOGGER_LEVEL", logging.WARNING))
logger = logging.getLogger(__name__)
    
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

