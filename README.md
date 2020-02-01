# numerai_train

This repository contains code to train, tune, and make/submit predictions using the Numerox API to the Numerai Data Science Tournament.

https://docs.numer.ai/tournament/learn

# Setup:

To train the models or make predictions I recommend that you first create a virtual environment and install the requirements file.

### Using Virtualenv

Use Python 3.7 to avoid some import issues I encountered:
- `virtualenv --python=/usr/bin/python3.7 <path/to/new/virtualenv/>`
- On Posix systems: `source /path/to/ENV/bin/activate`
- On Windows (where this repo was developed): `\path\to\env\Scripts\activate`

### Using conda

- `conda create -n yourenvname python=x.x anaconda`
- `source activate yourenvname`

### Install requirements

Once you have your virtual environment set up, install the requirements:
- `pip install -r requirements.txt`

# Training/Inference:

The API has a few basic components that come together in the `predict.py` module. There are Models and Trainers that compose a `train_and_predict_model()` function within the predict module. A Model is built on top of the Numerox API to make submission a bit simpler; a Trainer contains functionality to train, load, save models and also submit predictions - locally or from/to an s3 bucket.

1. Create a parameter dictionary:
- `EXAMPLE_PARAMS = {
            'depth': 7,
            'learning_rate': 0.1,
            'l2': 0.01,
            'iterations': 100
        }`
- Use parameter dictionary as an argument to `params=EXAMPLE_PARAMS` to the `train_and_predict_<model-name>_model()` function.

3. To then train, make predictions, and submit predictions run the following command. After training is complete the model weights will be saved to disk with the model name attached to it like this: `<model-name>_model_trained_<competition-name>`. You also have the option of saving the model to an s3 bucket (more on that below):
- `python predict.py --model <model-name> --load-model <bool> --save-model <bool> --submit <bool>`


# Saving model to s3 bucket:

1. Setup ennvironment variables for AWS s3 bucket:
- `export BUCKET=<bucket-name>`
- `export AWS_ACCESS_KEY_ID=<access-key-id>`
- `export AWS_SECRET_KEY=<secret-key>` 

2. Default setup is for models to be loaded to and from an s3 bucket, so run `predict.py` module as is. If you want to also save the models locally change the code by calling `trainer.save/load_model_locally()` methods.

# Submitting predictions:

1. Setup ennvironment variables for NumerAPI:
- `export NUMERAI_PUBLIC_ID=<public-id>`
- `export NUMERAI_SECREY_KEY=<secret-key>`

2. Set the `--submit` parameter to True when running `python predict.py --submit`

# Running Tests

With `numerai_train` as your working directory, run the following from the command line:
- `python -m pytest tests/tests_unit.py -v`

# Training models using AWS ECS

WIP

# Running experiments on Polyaxon

- `polyaxon login --username=root --password=rootpassword`
- `polyaxon project create --name=numerai_training --description='Train models on polyaxon'`
- `polyaxon init numerai_training`
- CPU: `polyaxon run -f configs/polyaxon_cpu.yaml`
- GPU: `polyaxon run -f configs/polyaxon_gpu.yaml`
