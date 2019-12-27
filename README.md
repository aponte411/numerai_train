# numerai_train

This repo contains code to train Numerai models and make submissions to the Numerai competition.
https://docs.numer.ai/tournament/learn

# Setup

To train the models or make predictions I recommend that you first create a virtual environment and install the requirements file

### Using Virtualenv

Use Python 3.7 to avoid some import issues I encountered:
- `virtualenv --python=/usr/bin/python3.7 <path/to/new/virtualenv/>`
- On Posix systems: `source /path/to/ENV/bin/activate`
- On Windows (where this repo was developed): `\path\to\env\Scripts\activate`

### Using conda

- `conda create -n yourenvname python=x.x anaconda`
- `source activate yourenvname`

# Training the models

Once you have your virtual environment set up, install the requirements:
- `pip install -r requirements.txt`

To train models run the following command. After training is complete the model weights will be saved to disk with the model name attached to `<model-name>_model_trained_<competition-name>`:
- `python train.py --model <model-name>`

# Making predictions

To train and make predictions run the following command:
`python predict --model <model-name>`


# Running Tests

With `numerai_train` as your working directory, run the following from the command line:
- `python -m pytest tests\tests_unit.py -v`