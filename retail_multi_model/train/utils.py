import os
import numpy as np
from PIL import Image

def load_images_with_labels_from_folder(folder):
    images = []
    labels = []
    # Loop through each folder inside the main folder
    for folder_name in os.listdir(folder):
        # Check if the folder is a directory
        if os.path.isdir(os.path.join(folder, folder_name)):
            # Loop through each image inside the folder
            for image_name in os.listdir(os.path.join(folder, folder_name)):
                # Check if the image is a file and ends with .png
                if os.path.isfile(os.path.join(folder, folder_name, image_name)) \
                    and image_name.endswith(".png"):
                    # Load the image and the label
                    image = Image.open((os.path.join(folder, folder_name, image_name)))
                    label = folder_name
                    # Add the image and the label to the list
                    images.append(image)
                    labels.append(label)
                    image.close()
    return images, labels

