import os
import numpy as np
from PIL import Image
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from retail_multi_model.train.dataset import FreiburgGroceriesDataset

logging.basicConfig(level=os.getenv("LOGGER_LEVEL", logging.WARNING))
logger = logging.getLogger(__name__)

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
    # Labels to categorical
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    # Log length
    logger.info("Loaded %d images", len(images))
    return images, labels

def train_using_tensorflow(train, test, labels_train, labels_test):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

    # Torch tensor to numpy array
    train = train.data['pixel_values'].detach().numpy()
    test = test.data['pixel_values'].detach().numpy()
    train = train.reshape(train.shape[0], train.shape[2], train.shape[3], 3)
    test = test.reshape(test.shape[0], test.shape[2], test.shape[3], 3)
    print(train.shape)

    shape = train.shape[1:]

    # Import VGG16
    from tensorflow.keras.applications.vgg16 import VGG16

    # base_model = VGG16(include_top=False,
    #               input_shape = shape,
    #               weights = 'imagenet')

    # for layer in base_model.layers:
    #     layer.trainable = False
        
    # for layer in base_model.layers:
    #     print(layer,layer.trainable)

    # model = Sequential()
    # model.add(base_model)
    # model.add(GlobalAveragePooling2D())
    # model.add(Dropout(0.3))
    # model.add(Dense(25,activation='softmax'))
    # model.summary()

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(25, activation='softmax'))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=['accuracy'])

    history = model.fit(train, labels_train, epochs=30, validation_data=(test, labels_test))

    # Save history as csv
    import pandas as pd
    pd.DataFrame(history.history).to_csv('history.csv')
    

def prepare_dataset(images,
                    labels,
                    model,
                    test_size=.2,
                    train_transform=None,
                    val_transform=None):
    # Split the dataset in train and test
    images_train, images_test, labels_train, labels_test = \
        train_test_split(images, labels, test_size=test_size)

    # Convert pillow images to numpy arrays
    # images_train = np.array([np.array(image) for image in images_train])
    # images_test = np.array([np.array(image) for image in images_test])

    # To float
    # images_train = images_train.astype('float32')
    # images_test = images_test.astype('float32')

    # # Preprocess the images
    # images_train /= 255
    # images_test /= 255

    # Preprocess images using model feature extractor
    print(type(images_train))
    print(type(images_train[0]))
    images_train = model.preprocess_image(images_train)
    images_test = model.preprocess_image(images_test)
    print(type(images_train))
    print(type(images_train['pixel_values']))

    # print(images_train.data['pixel_values'].shape)
    # train_using_tensorflow(images_train,
    #                        images_test,
    #                        labels_train,
    #                        labels_test)

    # Create the datasets
    train_dataset = FreiburgGroceriesDataset(images_train, labels_train, train_transform)
    test_dataset = FreiburgGroceriesDataset(images_test, labels_test, val_transform)
    return train_dataset, test_dataset