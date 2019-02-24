#ifdef __APPLE__

#define OSX

#define OPENREADOPTIONS O_RDONLY
#define OPENWRITEOPTIONS O_WRONLY|O_CREAT|O_TRUNC

#else

#define LINUX
#define _LARGEFILE_SOURCE 
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64

#define OPENREADOPTIONS O_RDONLY|O_LARGEFILE
#define OPENWRITEOPTIONS O_WRONLY|O_CREAT|O_TRUNC|O_LARGEFILE

#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "../vsib/vheader.h"

/* Invert ALL channels in a a VSIB data file. Currently this assume
   that the size of and "int" is 4 bytes (ie 32 bits). If not this
   program will fail. It works by XOR'ing the data from every second
   time sample with an appropriate mask for offset binary and
   sign/magnitude data. As it works on 32 bits at a time, it will
   currently not work with mode which use more that a total of 16bits
   of data (or numbers that don't divide into 32
*/

#include <unistd.h>
#include <getopt.h>
#include <limits.h>

#define BLOCKSIZE 4*1024*1024

#define RATE 32
#define BIGSTR 1024


int main (int argc, char * const argv[]) {
  int i, infile, outfile, nread, nwrote, opt, nchan;
  float bandwidth;
  unsigned int *data;
  unsigned char *c, lookup[256];
  char *mem, msg[256], fname[BIGSTR+1], oname[BIGSTR+1];
  off_t off;
  unsigned int mask, submask;
  encodingtype encoding;
  vhead *header;

  header = newheader();

  /* Command line arguments */
  encodingtype defaultencoding; /* VLBA or AT bit encoding (offset binary or sign/mag) */
  int verbose;      /* Prints whats happening */
  int shuffle;

  struct option options[] = {
    {"vlba", 0, 0, 'v'},
    {"at", 0, 0, 'a'},
    {"shuffle", 0, 0, 's'},
    {"verbose", 0, 0, 'V'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  if (sizeof(unsigned int)!=4) {
    fprintf(stderr, "This program assumes 32bit word size for the \"int\" type.\n");
    fprintf(stderr, "This does not seem to be the case!! A port to your "
	    "architecture is probably needed!!\n");
    exit(1);
  }

  verbose = 0;
  shuffle = 0;
  defaultencoding = NONE;

  /* Read command line options */
  while (1) {
    opt = getopt_long_only(argc, argv, "", options, NULL);
    if (opt==EOF) break;

    switch (opt) {
      
    case 's':
      shuffle = 1;
      break;

    case 'v':
      defaultencoding = VLBA;
      break;

    case 'a':
      defaultencoding = AT;
      break;

    case 'V':
      verbose = 1;
      break;

    case 'h':
      printf("Usage: band_invert [options] <inputfile> <outputfile>\n");
      printf("  -v/-vlba        VLBA bit encoding (offset binary)\n");
      printf("  -a/-at          AT bit encoding (sign/magnitude)\n");
      printf("  -h/-help        This list\n");
      return(1);
      break;

    case '?':
    default:
      break;
    }
  }

  if (argc-optind != 2) {
    fprintf(stderr, "Usange band_invert <inputfile> <outputfile>\n");
    exit(1);
  }

  mem = malloc(BLOCKSIZE);
  if (mem==NULL) {
    sprintf(msg, "Trying to allocate %d MB", BLOCKSIZE/1024/1024);
    perror(msg);
    return(1);
  }

  strcpy(fname, argv[optind]);
  infile = open(fname, OPENREADOPTIONS);
  if (infile==-1) {
    sprintf(msg, "Failed to open input file (%s)", fname);
    perror(msg);
    free(mem);
    return(1);
  }

  strcpy(oname, argv[optind+1]);
  outfile = open(oname, OPENWRITEOPTIONS,S_IRWXU);
  if (outfile==-1) {
    sprintf(msg, "Failed to open input file (%s)", oname);
    perror(msg);
    free(mem);
    return(1);
  }

  /* Read then write header */

  readheader(header, infile, NULL);  // Error check??

  if (header->numbits!=2) {
    fprintf(stderr, "Only works with 2 bit data\n");
    exit(1);
  }

  if (defaultencoding==NONE)
    encoding = header->encoding;
  else
    encoding = defaultencoding;

  if (encoding == VLBA) {
    submask = 0x3; /* 11 */
  } else {  /* AT */
    submask = 0x1; /* 01 */ 
  }

  nchan = header->nchan;
  bandwidth = header->bandwidth;

  if (bandwidth==64.0) nchan=1;

  if (8%nchan !=0) {
    fprintf(stderr, "%d channels does not fit into 32bit word. Cannot run\n", nchan);
  }

  // Build up submask for all channels
  for (i=0; i<nchan; i++) {
    submask |= (submask&0x3) << i*2;
  }

  mask = 0;
  for (i=0; i*2*nchan*2<32; i++) {
    mask |= submask << i*2*nchan*2; // Assumes 2 bit
  }

  if (shuffle) {
    for (i=0; i<256; i++) {
      lookup[i] = ((i>>2)&0x3) | ((i<<2)&0xC) | ((i>>2)&0x30) | ((i<<2)&0xC0);
    }
  }

  writeheader(header, outfile, NULL);


  while (1) {
    nread = read(infile, mem, BLOCKSIZE);

    if (nread==0) {
      break;
    } else if (nread==-1) {
      perror("Error reading file");
      return(1);
    }

    
    if (! nread%4) { /* Need to work on 32bit words */
      off = lseek(infile, -(nread%4), SEEK_CUR); 
      nread -= nread%4;
    }

    /* Invert the spectrum. Do this by XORing with a mask */
    data = (unsigned int*)mem;
    c = (unsigned char*)mem;
    for (i=0; i<nread; i+=4) {
      *data = *data ^ mask;  
      data++;
      if (shuffle) {
	*c = lookup[*c];
	c++;
	*c = lookup[*c];
	c++;
	*c = lookup[*c];
	c++;
	*c = lookup[*c];
	c++;
      }
    }
    
    nwrote = write(outfile, mem, nread);

    if (nwrote==-1) {
      perror("Error writing outfile");
      return(1);
    } else if (nwrote!=nread) {
      fprintf(stderr, "Warning: Did not write all bytes! (%d/%d)\n",
	      nwrote, nread);
    } 
  }

  close(outfile);
  close(infile);
  return(1);
}

