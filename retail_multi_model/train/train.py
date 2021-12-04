import os
import click
from retail_multi_model.train.utils import (
    load_images_with_labels_from_folder,
    prepare_dataset
)
from retail_multi_model.core.model import ViTForImageClassification

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

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

@click.command()
@click.option('--dataset_path', default='images/', help='Path to the dataset')
@click.option('--num_images', default=None, help='Number of images per class to load')
@click.option('--pretrained_model_name',
              default='google/vit-base-patch16-224',
              help='Name of the model')
@click.option('--num_epochs', default=1, help='Number of epochs')
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--learning_rate', default=0.001, help='Learning rate')
@click.option('--image_size', default=224, help='Image size')
@click.option('--dropout_rate', default=0.5, help='Dropout rate')
def train(
        dataset_path,
        num_images,
        pretrained_model_name,
        num_epochs,
        batch_size,
        learning_rate,
        image_size,
        dropout_rate
    ):
    images, labels = load_images_with_labels_from_folder(dataset_path, num_images)
    model = ViTForImageClassification(
        model_name=pretrained_model_name,
        num_labels=len(set(labels)))

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
            eval_steps=100,
            save_steps=100),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train(resume_from_checkpoint='output/checkpoint-12400')
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    # Evaluate the model
    eval_result = trainer.evaluate()
    # trainer.save_metrics("eval", eval_result.metrics)
    print(eval_result)
    # Save eval_result to a file
    with open('eval_result.txt', 'w') as f:
        f.write(str(eval_result))
    # Create directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    model.save('model/')


if __name__ == '__main__':
    train()