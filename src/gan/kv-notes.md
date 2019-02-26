# lba.py

1. Line 132.
I've included the evlbi software from CSIRO. 
The values for VLBA are different in some files. 
I guess it comes down to which we believe.  
    * fauto.c, vsib_record.c have it as +3, -1, +1, -3
    * fcross.c has it as +3, +1, -1, -3
    * vsib_checker seems to have it as +3, +1, -1, -3
    
2. Line 248.
I'm not convinced the polarisation is being extracted correctly. 
The vex file from the observation shows something different  
    
# preprocess.py

1. I've symbolically linked the lba.py and jobs.py filed into the GAN so we don't have to dynamic change the **PYTHONPATH**
2. Line 143 & 148. **Why take the 1/2 the real and amplitude and concatenate it with all the imaginary**   

