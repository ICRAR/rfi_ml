"""
Contains the preprocessor for converting input files into the common HDF5 formats.

Data preparation follows the following steps:

1. Convert a raw data file into an `src.preprocess.hdf5_definition.HDF5Observation` file with a common format. This file
   contains the raw samples from the input file along with observation metadata if applicable.
   This step is performed by `src.preprocess.main`.

2. Convert the `src.preprocess.hdf5_definition.HDF5Observation` file into a `src.preprocess.fft.hdf5_fft_definition.HDF5FFTDataSet` file.
   This file contains GAN inputs, where each input is the absolute and angle values from the FFT of a segment of samples
   from the `src.preprocess.hdf5_definition.HDF5Observation` file. This step is performed by `src.preprocess.fft.main`.

3. Load the data into the GAN using the `src.data.Data` loader and use it for training.
"""