# Preprocessing

This directory contains both the raw and preprocessed machine learning data. 
The files are too large for git so they need to be copied in manually.

All of the preprocessing scripts will create two files
- **\*.hdf5** containing the raw samples from the input files along with some associated metadata.
- **\*_fft.hdf5** containing the FFTs of the raw samples to be used for GAN input.

## Powerline Data
Extract the archive into the `raw` directory into the folder `power_line_RFI_data`.
The directory structure should be as follows
- raw
  - power_line_RFI_data
    - C148700000.txt
    - C148700001.txt
    - C148700002.txt
    
Next, run the `preprocess_india_txt.sh` script to preprocess the data. The following files should be created in the `processed` directory.

- C148700000.hdf5
- C148700001.hdf5
- C148700002.hdf5
- C148700000_fft.hdf5
- C148700001_fft.hdf5
- C148700002_fft.hdf5

## VLBA Data
Create the `vlba` directory inside the `raw` directory and copy in the following files:

- v255ae.vex
- v255ae_At_072_060000.lba
- v255ae_Mp_072_060000.lba
- vt255ae_Pa_072_060000.lba

Next, run the `preprocess_vlba.sh` script to preprocess the data. The following files should be created in the `processed` directory.

- v255ae_At_072_060000.hdf5
- v255ae_Mp_072_060000.hdf5
- vt255ae_Pa_072_060000.hdf5
- v255ae_At_072_060000_fft.hdf5
- v255ae_Mp_072_060000_fft.hdf5
- vt255ae_Pa_072_060000_fft.hdf5