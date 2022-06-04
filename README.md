# Image-Classification-of-Retail-Products

Leverages a multi class classification model for classifying retail products from a supermarket.

## Install

* Install poetry in your OS using [this guide](https://python-poetry.org/docs/)
* Get into the env `poetry shell`
* Install env `poetry install`
* Run `poe force-cuda11` to install the required torch CUDA version

## Training

For training, I use ImagineS, a library of my own, to do scrapping of images on Google Search Engine to create a dataset for Grocery Classification.
In order to train the model, we need to:

* Install chrome in your machine.
* Make sure `.data/label_queries.json` has the labels and queries you need.
* Review training parameters in `python retail_multi_model/train/train.py`:
  * `download_images_path`: Path to download images to.
  * `num_images`: Number of images per class to load.
  * `pretrained_model_name`: Name of the pretrained model to use.
  * `num_epochs`: Number of epochs to train for.
  * `batch_size`: Batch size to use.
  * `learning_rate`: Learning rate to use.
  * `image_size`: Size of the images to use. If images come in different sizes, the images will be resized to this size.
  * `dropout`: Dropout of the last layer to use.
  * `last_checkpoint_path`: Path to the last checkpoint to use. Default is None.
* Execute `python retail_multi_model/train/train.py` for training, adding the values of the parameters that you need.
* Resulting checkpoints and the final model will be saved in `output/`.

## Server & Interface

1. Set the corresponding env variables in `env_vars.env` file. Current env variables are:

   * MODEL_PATH: Path to the model.
   * SERVER_PORT: Port to run the server from.
   * INTERFACE_PORT: Port to run the interface from.

2. Execute server and interface:

    * `sh run_server_interface.sh`

3. Open the browser and navigate to `http://localhost:$INTERFACE_PORT/`.

4. Load a picture of a product and get a prediction.
