# rfi_ml
Machine learning code for RFI 

All of the scripts in this file must be run from the root of the repository.

## Installation
Virtual Env installation
```bash
# Create venv and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r scripts/requirements.txt

# Install pyvex for reading .vex files
cd pyvex/pyvex
python setup.py install
```

## Preprocessing
```bash
# Preprocess power line data
bash data/preprocess_india_txt.sh

# Preprocess vlba data
bash data/preprocess_vlba.sh
```

## Documentation Generation
```bash
source venv/bin/activate
pdoc3 --html src --force
```
The documentation will be available in `html` for browsing.
Additionally, documentation is written in [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html#overview) format

## Configuration
Default config file creation
```bash
source venv/bin/activate
python -m src.config
```
The config file will be created in the root of the repository

## Training
```bash
source venv/bin/activate
# Train using the config file in the root
python -m src.train gan_config.settings
```

**Configuration Options**

`USE_CUDA` - True to train using the GPU, false to use the CPU (defaults to true).

`FILENAME` - Path to HDF5 file to load data from, relative to the repository root.

`MAX_EPOCHS` - Max number of epochs to train the GAN for (defaults to 60).

`MAX_GENERATOR_AUTOENCODER_EPOCHS` - Max number of epochs to train the generator autoencoder for (defaults to 60).

`MAX_SAMPLES` - Maximum number of inputs to train on. Set to 0 for unlimited (defaults to 0).

`BATCH_SIZE` - Number of samples to train on per batch (defaults to 4096).

`POLARISATIONS` - Which polarisations should be used? (comma separated list, defaults to 0, 1).

`FREQUENCIES` - Which frequencies should be used? (comma separated list, defaults to 0, 1, 2, 3).

`NORMALISE` - Set to true to normalise inputs (defaults to true).

`ADD_DROPOUT` - If true, add dropout to the inputs before passing them into the network (defaults to true).

`ADD_NOISE` - If true, add noise to the inputs before passing them into the network (defaults to false).

`REQUEUE_EPOCHS` - If > 0, perform REQUEUE_EPOCHS of training, stop, then run the REQUEUE_SCRIPT (defaults to 0).

`REQUEUE_SCRIPT` - If REQUEUE_EPOCHS > 0, this script will be called to requeue the training script.

`CHECKPOINT_DIRECTORY` - The directory to write checkpoints to, relative to the repository root.

`RESULT_DIRECTORY` - The directory to write results to, relative to the repository root.

**Example**
```text
USE_CUDA = True
FILENAME = data/processed/v255ae_At_072_060000_fft.hdf5
MAX_EPOCHS = 60
MAX_AUTOENCODER_EPOCHS = 60
MAX_SAMPLES = 0
BATCH_SIZE = 128
NORMALISE = True
ADD_DROPOUT = False
ADD_NOISE = False
REQUEUE_EPOCHS = 0
REQUEUE_SCRIPT = ""
CHECKPOINT_DIRECTORY = data/checkpoints
RESULT_DIRECTORY = data/results/
```
