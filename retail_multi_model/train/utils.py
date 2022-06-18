import os
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from retail_multi_model.train.dataset import RetailDataset
from transformers import BatchFeature
from PIL import Image
from datasets import load_metric
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import Trainer, TrainingArguments
metric = load_metric("accuracy")
f1_score = load_metric("f1")
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
    try:
        images_train, images_test, labels_train, labels_test = \
            train_test_split(images, labels, test_size=test_size)
    except ValueError:
        logger.warning("Could not split dataset. Using all data for training and testing")
        images_train = images
        labels_train = labels
        images_test = images
        labels_test = labels

    # Preprocess images using model feature extractor
    images_train_prep = []
    images_test_prep = []
    for bs in tqdm(range(0, len(images_train), batch_size), desc="Preprocessing training images"):
        images_train_batch = [Image.fromarray(np.array(image)) for image in images_train[bs:bs+batch_size]]
        images_train_batch = model.preprocess_image(images_train_batch)
        images_train_prep.extend(images_train_batch['pixel_values'])
    for bs in tqdm(range(0, len(images_test), batch_size), desc="Preprocessing test images"):
        images_test_batch = [Image.fromarray(np.array(image)) for image in images_test[bs:bs+batch_size]]
        images_test_batch = model.preprocess_image(images_test_batch)
        images_test_prep.extend(images_test_batch['pixel_values'])

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

def re_training(images, labels, _model, save_model_path='new_model', num_epochs=10):
    global model
    model = _model
    labels = model.label_encoder.transform(labels)
    normalize = Normalize(mean=model.feature_extractor.image_mean, std=model.feature_extractor.image_std)
    def train_transforms(batch):
        return Compose([
            RandomResizedCrop(model.feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])(batch)

    def val_transforms(batch):
        return Compose([
            Resize(model.feature_extractor.size),
            CenterCrop(model.feature_extractor.size),
            ToTensor(),
            normalize,
        ])(batch)
    train_dataset, test_dataset = prepare_dataset(
        images, labels, model, .2, train_transforms, val_transforms)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir='output',
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=1,
            learning_rate=0.000001,
            weight_decay=0.01,
            evaluation_strategy='steps',
            eval_steps=1000,
            save_steps=3000),
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    trainer.train()
    model.save(save_model_path)