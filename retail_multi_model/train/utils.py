import io
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from retail_multi_model.train.dataset import FreiburgGroceriesDataset
import requests
import hashlib
import time
from selenium import webdriver
np.random.seed(42)

logging.basicConfig(level=os.getenv("LOGGER_LEVEL", logging.WARNING))
logger = logging.getLogger(__name__)

op = webdriver.ChromeOptions()
op.add_argument('headless')
driver = webdriver.Chrome('./chromedriver',options=op)

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
            target_folder = search_and_download(q, driver, label, num_images)
            resize_images_from_folder(target_folder, (image_size, image_size))

def fetch_image_urls(query:str, wd:webdriver, max_links_to_fetch:int=10, sleep_between_interactions:float=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)    
    
    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)
        
        # print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        
        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls    
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls

def persist_image(folder_path:str, url:str):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        # print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")

def search_and_download(search_term:str, wd:webdriver, label:str, number_images=5, target_path='./new_images'):
    target_folder = os.path.join(target_path, label)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    res = fetch_image_urls(search_term, wd, number_images, sleep_between_interactions=0.5)
        
    for elem in res:
        persist_image(target_folder,elem)

    return target_folder

def resize_images_from_folder(folder_path:str, size:tuple):
    for file in os.listdir(folder_path):
        try:
            img = Image.open(os.path.join(folder_path, file))
            img = img.resize(size)
            img.save(os.path.join(folder_path, file))
        except Exception as e:
            print(e)