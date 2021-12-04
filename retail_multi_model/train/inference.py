from retail_multi_model.core.model import ViTForImageClassification
from retail_multi_model.train.utils import load_images_with_labels_from_folder

images, labels = load_images_with_labels_from_folder('images/', 5)

model = ViTForImageClassification('google/vit-base-patch16-224', 25)
model.load('pytorch_model/model.pth')
print(labels[0])
predictions = model.predict([images[0]])
print(predictions)