# rfi_ml
Machine learning code for RFI 

## Installation
Virtual Env installation
```bash
# if virtualenv is not already installed
pip install virtualenv

# Create venv and install dependencies
virtualenv venv
source venv/bin/activate
pip install scripts/requirements.txt
```

## Documentation Generation
```bash
source venv/bin/activate
pdoc3 --html src --force
```
The documentation will be available in `html` for browsing.



## Configuration
`config.py` contains the GAN configuration file parser. This script can be run directly to produce a default `gan_config.settings` file containing a default configuration.

**Configuration Options**

`USE_CUDA` - True to train using the GPU, false to use the CPU (defaults to true).

`FILENAME` - Path to HDF5 file to load data from.

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

**Example**
```text
USE_CUDA = True
FILENAME = ../data/processed/v255ae_At_072_060000_fft.hdf5
MAX_EPOCHS = 60
MAX_AUTOENCODER_EPOCHS = 60
MAX_SAMPLES = 0
BATCH_SIZE = 128
NORMALISE = True
ADD_DROPOUT = False
ADD_NOISE = False
REQUEUE_EPOCHS = 0
REQUEUE_SCRIPT = ./requeue.sh
CHECKPOINT_DIRECTORY = ../data/checkpoints
RESULT_DIRECTORY = ../data/results/

```
