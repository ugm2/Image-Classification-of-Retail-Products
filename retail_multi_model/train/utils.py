import io
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from retail_multi_model.train.dataset import RetailDataset
import hashlib
import time
from imagines import DatasetAugmentation
from transformers import BatchFeature
import torch
np.random.seed(42)

logging.basicConfig(level=os.getenv("LOGGER_LEVEL", logging.WARNING))
logger = logging.getLogger(__name__)
    
def prepare_dataset(images,
                    labels,
                    model,
                    test_size=.2,
                    batch_size=16,
                    train_transform=None,
                    val_transform=None):
    logger.info("Preparing dataset")
    # Split the dataset in train and test
    images_train, images_test, labels_train, labels_test = \
        train_test_split(images, labels, test_size=test_size)

    # Preprocess images using model feature extractor
    images_train_prep = images_train.copy()
    images_test_prep = images_test.copy()
    for bs in tqdm(range(0, len(images_train), batch_size), desc="Preprocessing training images"):
        images_train_batch = images_train[bs:bs+batch_size]
        images_train_batch = model.preprocess_image(images_train_batch)
        images_train_prep[bs:bs+batch_size] = images_train_batch['pixel_values']
    for bs in tqdm(range(0, len(images_test), batch_size), desc="Preprocessing test images"):
        images_test_batch = images_test[bs:bs+batch_size]
        images_test_batch = model.preprocess_image(images_test_batch)
        images_test_prep[bs:bs+batch_size] = images_test_batch['pixel_values']
    
    # Create BatchFeatures
    # print(len(images_test_prep))
    # print(type(images_test_prep[0]))
    train_batch_features = BatchFeature({'pixel_values': np.array(images_train_prep)})
    test_batch_features = BatchFeature({'pixel_values': np.array(images_test_prep)})

    # Create the datasets
    train_dataset = RetailDataset(train_batch_features, labels_train, train_transform)
    test_dataset = RetailDataset(test_batch_features, labels_test, val_transform)
    logger.info("Train dataset: %d images", len(labels_train))
    logger.info("Test dataset: %d images", len(labels_test))
    return train_dataset, test_dataset

