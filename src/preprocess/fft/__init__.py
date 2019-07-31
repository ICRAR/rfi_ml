"""
Preprocessing step 2: Perform an FFT over the preprocessed data and save in common format.

- This process should extracts multiples of FFT size chunks (e.g. fft x 10) from the input file
  and place them onto a processing queue.
- FFT calculation processes are spawned and accept items from the queue, perform the FFT on the items,
  then place the result back on a return queue.
- This process accepts items from the destination queue (items here contain an index as to which FFT they are)
  and writes them out to the HDF5 file.
"""