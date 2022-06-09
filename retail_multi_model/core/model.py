import numpy as np
from transformers import ViTModel, ViTFeatureExtractor
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch

import logging
import os
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=os.getenv("LOGGER_LEVEL", logging.WARNING))
logger = logging.getLogger(__name__)

class ViTForImageClassification(nn.Module):
    def __init__(self, model_name, num_labels=25, dropout=0.25, image_size=224):
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

    def predict(self, images):
        logger.info("Predicting")
        prep_images = self.preprocess_image(images)['pixel_values']
        sequence_classifier_output = self.forward(prep_images, None)
        # Get max prob
        probs = sequence_classifier_output.logits.softmax(dim=-1).tolist()
        class_nums = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        class_names = self.label_encoder.inverse_transform(class_nums)
        return class_names, confidences

    def save(self, path):
        logger.info("Saving model")
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/model.pt")
        # Save label encoder
        np.save(path + "/label_encoder.npy", self.label_encoder.classes_)

    def load(self, path):
        logger.info("Loading model")
        self.load_state_dict(torch.load(path + "/model.pt"))
        # Load label encoder
        self.label_encoder.classes_ = np.load(path + "/label_encoder.npy")
        self.vit.to(self.device)
        self.vit.eval()
        self.to(self.device)
        self.eval()
