import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from retail_multi_model.train.dataset import FreiburgGroceriesDataset
from google_images_search import GoogleImagesSearch
np.random.seed(42)

logging.basicConfig(level=os.getenv("LOGGER_LEVEL", logging.WARNING))
logger = logging.getLogger(__name__)

gis = GoogleImagesSearch('AIzaSyBS1XZKJLmE6lud4j8nWvxkSNXw1RwZBuw', 'e643bed31b3b0443b')

def load_images_with_labels_from_folder(folder, num_images=None):
    images = []
    labels = []
    # Loop through each folder inside the main folder
    for folder_name in tqdm(os.listdir(folder), desc='Loading images'):
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
                    images.append(image.copy())
                    labels.append(label)
                    image.close()
                    # Check if the number of images is reached
                    if num_images is not None and len(images) >= num_images:
                        break
    # Log length
    logger.info("Loaded %d images", len(images))
    return images, labels
    

def prepare_dataset(images,
                    labels,
                    model,
                    test_size=.2,
                    train_transform=None,
                    val_transform=None):
    logger.info("Preparing dataset")
    # Split the dataset in train and test
    images_train, images_test, labels_train, labels_test = \
        train_test_split(images, labels, test_size=test_size)

    # Preprocess images using model feature extractor
    images_train = model.preprocess_image(images_train)
    images_test = model.preprocess_image(images_test)

    # Create the datasets
    train_dataset = FreiburgGroceriesDataset(images_train, labels_train, train_transform)
    test_dataset = FreiburgGroceriesDataset(images_test, labels_test, val_transform)
    logger.info("Train dataset: %d images", len(labels_train))
    logger.info("Test dataset: %d images", len(labels_test))
    return train_dataset, test_dataset

def augment_dataset(labels, num_images_per_class=10, image_size=224):
    logger.info("Augmenting dataset")
    labels_set = set(labels)
    for label in tqdm(labels_set, desc='Augmenting dataset'):
        print(label)
        if label == 'BEANS':
            query = ['white beans', 'black beans', 'red beans']
            query.append([q + ' can' for q in query])
        elif label == 'CAKE':
            query = ['cake', 'cake grocery store', 'dry cake', 'muffins packet']
        elif label == 'CANDY':
            query = ['candy packet', 'sweets candy', 'sweets supermarket']
        elif label == 'CEREAL':
            query = ['cereal', 'cereal packet']
        elif label == 'CHIPS':
            query = ['chips', 'chips packet']
        elif label == 'CHOCOLATE':
            query = ['chocolate packet']
        elif label == 'COFFEE':
            query = ['coffee grocery', 'coffee packet']
        elif label == 'CORN':
            query = ['corn can', 'corn packet']
        elif label == 'FISH':
            query = ['fish can', 'fish packet']
        elif label == 'FLOUR':
            query = ['flour packet']
        elif label == 'HONEY':
            query = ['honey grocery', 'honey jar']
        elif label == 'JAM':
            query = ['jam jar']
        elif label == 'JUICE':
            query = ['juice bottle', 'juice packet']
        elif label == 'MILK':
            query = ['milk bottle', 'milk packet']
        elif label == 'NUTS':
            query = ['nuts packet']
        elif label == 'OIL':
            query = ['oil bottle']
        elif label == 'PASTA':
            query = ['pasta packet']
        elif label == 'RICE':
            query = ['rice packet']
        elif label == 'SODA':
            query = ['soda bottle', 'soda can']
        elif label == 'SPICES':
            query = ['spices bottle']
        elif label == 'SUGAR':
            query = ['sugar packet']
        elif label == 'TEA':
            query = ['tea packet', 'tea bags']
        elif label == 'TOMATO_SAUCE':
            query = ['tomato sauce', 'tomato sauce packet', 'tomato sauce can']
        elif label == 'VINEGAR':
            query = ['vinegar bottle', 'vinegar can', 'vinegar packet']
        elif label == 'WATER':
            query = ['water bottle', 'water bottle supermarket']
        num_images = round(num_images_per_class / len(query))
        for q in query:
            google_image_search(q, num_images, label, image_size)

def google_image_search(query, num_images=10, download_path=None, image_size=224):
    # define search params:
    _search_params = {
        'q': query,
        'num': num_images,
        'fileType': 'jpg | png',
        'imgSize': 'LARGE',
        'imgColorType': 'color'
    }
    if download_path is None:
        download_path = os.path.join(os.getcwd(), 'new_images') + f'/{query}'
    else:
        download_path = os.path.join(os.getcwd(), 'new_images') + f'/{download_path}'
    gis.search(search_params=_search_params,
               path_to_dir=download_path,
               width=image_size,
               height=image_size)

google_image_search('beans')