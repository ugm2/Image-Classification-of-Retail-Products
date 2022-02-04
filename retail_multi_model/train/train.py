import os
import click
from retail_multi_model.core.model import ViTForImageClassification
from retail_multi_model.train.utils import prepare_dataset

from transformers import Trainer, TrainingArguments
from datasets import load_metric
import numpy as np
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from imagines import DatasetAugmentation

metric = load_metric("accuracy")
f1_score = load_metric("f1")

import pandas as pd

metrics_list = []

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = metric.compute(predictions=predictions, references=labels)
    f1 = f1_score.compute(predictions=predictions, references=labels, average="macro")
    metrics.update(f1)
    metrics_list.append(metrics)
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv("metrics.csv")
    return metrics

@click.command()
@click.option('--download_images_path', default='data', help='Path where to download dataset')
@click.option('--num_images', default=1200, help='Number of images per class to load')
@click.option('--pretrained_model_name',
              default='google/vit-base-patch16-224',
              help='Name of the model')
@click.option('--num_epochs', default=20, help='Number of epochs')
@click.option('--batch_size', default=64, help='Batch size')
@click.option('--learning_rate', default=0.000001, help='Learning rate')
@click.option('--image_size', default=224, help='Image size')
@click.option('--dropout', default=0.5, help='Dropout rate')
def train(
        download_images_path,
        num_images,
        pretrained_model_name,
        num_epochs,
        batch_size,
        learning_rate,
        image_size,
        dropout
    ):

    # Load the dataset
    images, labels = DatasetAugmentation().augment_dataset(
        label_queries=os.path.join(download_images_path, "label_queries.json"),
        output_directory=os.path.join(download_images_path, "images"),
        max_links_to_fetch=num_images,
        image_shape=(image_size, image_size),
        resize_images=True,
        return_data=True,
        cache_data=True
    )

    model = ViTForImageClassification(
        model_name=pretrained_model_name,
        num_labels=len(set(labels)),
        dropout=dropout)

    # Labels to categorical
    labels = model.label_encoder.fit_transform(labels)

    # Define torchvision transforms to be applied to each image.
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
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            weight_decay=0.01,
            evaluation_strategy='steps',
            eval_steps=1000,
            save_steps=3000),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    # train_result = trainer.train(resume_from_checkpoint='output/checkpoint-6000')
    # Evaluate the model
    eval_result = trainer.evaluate()
    print(eval_result)
    model.save('model')
    log_history_df = pd.DataFrame(trainer.state.log_history)
    log_history_df.to_csv("log_history.csv")


if __name__ == '__main__':
    train()