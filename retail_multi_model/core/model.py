import shutil
import time
import numpy as np
from tqdm import tqdm
from transformers import ViTModel, ViTFeatureExtractor
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch
from PIL import Image
import logging
import os
from sklearn.preprocessing import LabelEncoder
from retail_multi_model.train.utils import (
    re_training, metric, f1_score,
    classification_report
)

data_path = os.environ.get('DATA_PATH', "./data")

logging.basicConfig(level=os.getenv("LOGGER_LEVEL", logging.WARNING))
logger = logging.getLogger(__name__)

class ViTForImageClassification(nn.Module):
    def __init__(self, model_name, num_labels=24, dropout=0.25, image_size=224):
        logger.info("Loading model")
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.feature_extractor.do_resize = True
        self.feature_extractor.size = image_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels
        self.label_encoder = LabelEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        # To device
        self.vit.to(self.device)
        self.to(self.device)
        self.classifier.to(self.device)

    def forward(self, pixel_values, labels):
        logger.info("Forwarding")
        pixel_values = pixel_values.to(self.device)
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def preprocess_image(self, images):
        logger.info("Preprocessing images")
        return self.feature_extractor(images, return_tensors='pt')

    def predict(self, images, batch_size=32, classes_names=True, return_probabilities=False):
        logger.info("Predicting")
        if not isinstance(images, list):
            images = [images]
        classes_list = []
        confidence_list = []
        for bs in tqdm(range(0, len(images), batch_size), desc="Preprocessing training images"):
            images_batch = [image for image in images[bs:bs+batch_size]]
            images_batch = self.preprocess_image(images_batch)['pixel_values']
            sequence_classifier_output = self.forward(images_batch, None)
            # Get max prob
            probs = sequence_classifier_output.logits.softmax(dim=-1).tolist()
            classes = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)
            classes_list.extend(classes)
            confidence_list.extend(confidences)
        if classes_names:
            classes_list = self.label_encoder.inverse_transform(classes_list)
        if return_probabilities:
            return classes_list, confidence_list, probs
        return classes_list, confidence_list

    def save(self, path):
        logger.info("Saving model")
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/model.pt")
        # Save label encoder
        np.save(path + "/label_encoder.npy", self.label_encoder.classes_)

    def load(self, path):
        logger.info("Loading model")
        # Load label encoder
        # Check if label encoder and model exists
        if not os.path.exists(path + "/label_encoder.npy") or not os.path.exists(path + "/model.pt"):
            logger.warning("Label encoder or model not found")
            return
        self.label_encoder.classes_ = np.load(path + "/label_encoder.npy")
        # Reload classifier layer
        self.classifier = nn.Linear(self.vit.config.hidden_size, len(self.label_encoder.classes_))
        
        self.load_state_dict(torch.load(path + "/model.pt", map_location=self.device))
        self.vit.to(self.device)
        self.vit.eval()
        self.to(self.device)
        self.eval()
        
    def evaluate(self, images, labels):
        logger.info("Evaluating")
        labels = self.label_encoder.transform(labels)
        # Predict
        y_pred, _ = self.predict(images, classes_names=False)
        # Evaluate
        metrics = metric.compute(predictions=y_pred, references=labels)
        f1 = f1_score.compute(predictions=y_pred, references=labels, average="macro")
        print(classification_report(labels, y_pred, labels=[i for i in range(len(self.label_encoder.classes_))], target_names=self.label_encoder.classes_))
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"F1: {f1}")
        
    def partial_fit(self, images, labels, save_model_path='new_model', num_epochs=10):
        logger.info("Partial fitting")
        # Freeze ViT model but last layer
        # params = [param for param in self.vit.parameters()]
        # for param in params[:-1]:
        #     param.requires_grad = False
        # Model in training mode
        self.vit.train()
        self.train()
        re_training(images, labels, self, save_model_path, num_epochs)
        self.load(save_model_path)
        self.vit.eval()
        self.eval()
        self.evaluate(images, labels)
        
    def __load_from_path(self, path, num_per_label=None):
        images = []
        labels = []
        for label in os.listdir(path):
            count = 0
            label_folder_path = os.path.join(path, label)
            for image_file in tqdm(os.listdir(label_folder_path), desc="Resizing images for label {}".format(label)):
                file_path = os.path.join(label_folder_path, image_file)
                try:
                    image = Image.open(file_path)
                    image_shape = (self.feature_extractor.size, self.feature_extractor.size)
                    if image.size != image_shape:
                        image = image.resize(image_shape)
                    images.append(image.convert('RGB'))
                    labels.append(label)
                    count += 1
                except Exception as e:
                    print(f"ERROR - Could not resize image {file_path} - {e}")
                if num_per_label is not None and count >= num_per_label:
                    break
        return images, labels
        
    def retrain_from_path(self,
                          path='./data/feedback',
                          num_per_label=None,
                          save_model_path='new_model',
                          remove_path=False,
                          num_epochs=10,
                          save_new_data=data_path + '/new_data'):
        logger.info("Retraining from path")
        # Load path
        images, labels = self.__load_from_path(path, num_per_label)
        # Retrain
        self.partial_fit(images, labels, save_model_path, num_epochs)
        # Save new data
        if save_new_data is not None:
            logger.info("Saving new data")
            for i ,(image, label) in enumerate(zip(images, labels)):
                label_path = os.path.join(save_new_data, label)
                os.makedirs(label_path, exist_ok=True)
                image.save(os.path.join(label_path, str(int(time.time())) + f"_{i}.jpg"))
        # Remove path folder
        if remove_path:
            logger.info("Removing feedback path")
            shutil.rmtree(path)
        
    def evaluate_from_path(self, path, num_per_label=None):
        logger.info("Evaluating from path")
        # Load images
        images, labels = self.__load_from_path(path, num_per_label)
        # Evaluate
        self.evaluate(images, labels)
                    
