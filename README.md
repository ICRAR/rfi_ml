# rfi_ml
Machine learning code for RFI 

## Documentation Generation
```python
source venv/bin/activate
pip install pdoc3
pdoc3 --html src --force
```
The documentation will be available in `html` for browsing.

## GAN
The main code for pre-processing data and running the GAN can be found in the `gan` directory.

### Configuration
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
FILENAME = At.hdf5
MAX_EPOCHS = 60
MAX_GENERATOR_AUTOENCODER_EPOCHS = 5
MAX_SAMPLES = 0
BATCH_SIZE = 4096
POLARISATIONS = 0, 1
FREQUENCIES = 0, 1, 2, 3
NORMALISE = False
ADD_DROPOUT = True
ADD_NOISE = False
REQUEUE_EPOCHS = 1
REQUEUE_SCRIPT = ./requeue.sh
```

### Preprocessing
`preprocess.py` accepts an lba data file containing 2-bit encoded data, and produces an HDF5 file containing GAN input data.

**Arguments**

`file` - The LBA file to read data from (required)

`outfile` - The HDF5 file to write to. If this file already exists, it will not be overwritten (required)

`--fft_window` - Specifies the window size, in raw samples, of the FFT to run over the raw samples (optional, defaults to 2048 samples).

`--max_ffts` - Specifies the maximum number of FFTs to create from the lba file. Each FFT is a single GAN input, so this specifies the number of GAN inputs to create. Set to 0 to create as many inputs as possible from the lba file (optional, defaults to 0).

The preprocessor can also be used as its own module.

Example
```python
from gan.preprocess import Preprocessor

preprocessor = Preprocessor('file.lba', 'outfile.hdf5', fft_window=2048, max_ffts=1000)
preprocessor()
```

### Data Loading
`data.py` and `HDF5Dataset.py` are responsible for providing the pre-processed dataset to the GAN for training.

`noise.py` - Provides an on the fly gaussian noise dataset.

`data.py` - Provides an iterator over a specified HDF5 dataset, and an iterator over two noise datasets.

**Example**
```python
from gan.data import Data
from gan.models.single_polarisation_single_frequency import Generator, Discriminator

# dataset, data type, batch size
# kwargs are passsed to HDF5Dataset
loader = Data('dataset.hdf5', 'real_imag', 4096)

generator = Generator(4096)
discriminator = Discriminator(4096)

for step, (data, noise1, noise2) in enumerate(loader):
    # Train GAN on data, noise1, and noise 2
    output = generator(noise1)
    output2 = discriminator(data)
    # ...
```

### Model
`models/` directory contains python files defining the discriminator and generator.

**Example**
```python
from gan.models.single_polarisation_single_frequency import Discriminator, Generator

discriminator = Discriminator(4096)
generator = Generator(4096)
```

### Checkpointing
`checkpoint.py` contains code for checkpointing the GAN model, optimiser, and epoch state.

**Example Loading**
```python
from torch.optim import Adam
from gan.checkpoint import Checkpoint
from gan.models.single_polarisation_single_frequency import Generator

# Create a checkpoint that uses the directory 'generator'
# This creates a 'checkpoint_generator' directory in the current working directory.
checkpoint = Checkpoint('generator')

# Create the model and optimiser we'll load the checkpoint into
generator = Generator(1024)
optimiser = Adam(generator.parameters())

# Load the *latest* checkpoint from the 'checkpoint_generator' directory
if checkpoint.load():
    try:
        # Latest checkpoint loaded successfully and state dicts are valid
        generator.load_state_dict(checkpoint.module_state)
        optimiser.load_state_dict(checkpoint.optimiser_state)
        print("Generator loaded at epoch {0}".format(checkpoint.epoch))
    except RuntimeError:
        # State dicts don't match the model
        print("Saved state does not match current model")
else:
    print("No checkpoints to load")
```

**Example Saving**
```python
from torch.optim import Adam
from gan.checkpoint import Checkpoint
from gan.models.single_polarisation_single_frequency import Generator

# Create the model and optimiser we'll load the checkpoint into
generator = Generator(1024)
optimiser = Adam(generator.parameters())
epoch = 0

# Create the checkpoint in the 'checkpoint_generator' directory
Checkpoint('generator', generator.state_dict(), optimiser.state_dict(), epoch).save()
```

## Visualisation
`visualise.py` runs a multiprocess job queue that creates PDFs that display the NNs training progress over time, and sample outputs of the NN for each epoch.

**Example**
```python
from gan.visualise import Visualiser
from gan.data import Data
from gan.models.single_polarisation_single_frequency import Discriminator, Generator

loader = Data('dataset.hdf5', 'real_imag', 4096)

discriminator = Discriminator(4096)
generator = Generator(4096)

with Visualiser('output_path') as vis:
    # Provide new losses for the end of a training step.
    d_loss_real = 0
    d_loss_fake = 0
    g_loss = 0
    vis.step(d_loss_real, d_loss_fake, g_loss)
    
    # Create a training plot at the end of an epoch
    epoch = 0
    vis.plot_training(epoch)
    
    # Create plots of the generator and discriminator's output on the provided noise and data batches
    data, noise1, _ = iter(loader).__next__()
    vis.test(0, loader.get_input_size_first(), discriminator, generator, noise1, data)
```

### Training
`train.py` runs the actual training loop for the GAN and the Generator Autoencoder. 

The training process is interruptable; the model and optimiser states are saved to disk each epoch, and
the latest states are restored when the trainer is resumed.

If the script is not able to load the generator's saved state, it will attempt to load the pre-trained generator autoencoder
from the `generator_decoder_complete` checkpoint (if it exists). If this also fails, the generator is
pre-trained as an autoencoder. This training is also interruptable, and will produce the `generator_decoder_complete` checkpoint on completion.

On successfully restoring generator and discriminator state, the trainer will proceed from the earliest restored epoch.
For example, if the generator is restored from epoch 7 and the discriminator is restored from epoch 5, training will
proceed from epoch 5.

Visualisation plots are produces each epoch and stored in `/path_to_input_file_directory/{gan/generator_auto_encoder}/{timestamp}/{epoch}`

Each time the trainer is run, it creates a new `timestamp` directory using the current time.

**Example**
```python
from gan.train import Train
from gan.config import Config

config = Config('config_path.settings')
train = Train(config)
train()
```