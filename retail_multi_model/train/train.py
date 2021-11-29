import click
from retail_multi_model.train.utils import (
    load_images_with_labels_from_folder,
    prepare_dataset
)
from retail_multi_model.core.model import ViTForImageClassification

from transformers import Trainer, TrainingArguments

@click.command()
@click.option('--dataset_path', default='images/', help='Path to the dataset')
@click.option('--num_images', default=5, help='Number of images per class to load')
@click.option('--pretrained_model_name',
              default='google/vit-base-patch16-224-in21k',
              help='Name of the model')
@click.option('--num_epochs', default=10, help='Number of epochs')
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
    train_dataset, test_dataset = prepare_dataset(
        images, labels, model, test_size=.2)
    # print length of train and test datasets
    print(len(train_dataset))
    print(len(test_dataset))
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
            logging_dir='logs',
            logging_steps=10,
            save_steps=10),
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    trainer.train()

if __name__ == '__main__':
    train()