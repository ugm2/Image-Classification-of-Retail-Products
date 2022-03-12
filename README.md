# Image-Classification-of-Retail-Products

Leverages a multi class classification model for classifying retail products from a supermarket

## Download Dataset

* Execute `download_dataset.sh` to download the dataset
* A folder called `images/` will be created, containing the 25 different retail products

## Install

* Install poetry in your OS using [this guide](https://python-poetry.org/docs/)
* Install env `poetry install`
* Get into the env `poetry shell`

## Training

For training, we use ImagineS, a library of my own, to do scrapping of images on Google Search Engine to create a dataset for Grocery Classification.
In order to train the model, we need to:

* Install chrome in your machine.
* Make sure `.data/label_queries.json` has the labels and queries you need.
* Review training parameters in `python retail_multi_model/train/train.py` and execute.
* Resulting checkpoints and the final model will be saved in `output/`.

## Server & Interface

1. Set the corresponding env variables in `.env` file. Current env variables are:

   * MODEL_PATH: Path to the model.
   * SERVER_PORT: Port to run the server from.
   * INTERFACE_PORT: Port to run the interface from.

2. Execute server and interface:

    * `sh run_server_interface.sh`

3. Open the browser and navigate to `http://localhost:$INTERFACE_PORT/`.

4. Load a picture of a product and get a prediction.
