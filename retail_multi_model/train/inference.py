from retail_multi_model.core.model import ViTForImageClassification
from retail_multi_model.train.utils import load_images_with_labels_from_folder

images, labels = load_images_with_labels_from_folder('images/', 5)

model = ViTForImageClassification('google/vit-base-patch16-224')
model.load('model/')
print(labels)
predictions = model.predict(images)
print(predictions)