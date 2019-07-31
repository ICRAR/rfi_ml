#ifdef __APPLE__

#define OSX

#define OPENOPTIONS O_RDONLY

#else

#define LINUX
#define _LARGEFILE_SOURCE 
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64

#define OPENOPTIONS O_RDONLY|O_LARGEFILE

#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <limits.h>

#include "../vsib/vheader.h"

/***************************
 *        TODO             *
 ***************************

 - Implement "skip" option 

*/

#define BLOCKSIZE 4*1024*1024

#define MAXSTR 256

typedef enum {MODE_NONE, CHECK_1PPS, CHECK_COUNT, CHECK_STATS, 
	      SEARCH_PATTERN, CHECK_WIDEPPS} mode_type;

int main (int argc, char * const argv[]) {
  int nfile, i, j, file, status, nread, memsize, opt, first, ppsOK;
  int secount, tmp, psize, poverlap, headsize, bytes, widepps;
  unsigned int *data, diff, last, ppsskip, expected, wrap1, wrap2, pps;
  unsigned short *start;
  unsigned long *stats;
  char msg[MAXSTR+1], searchstr[MAXSTR+1];
  unsigned char  *ptr, *pattern, *mem;
  off_t off, offset;
  vhead *header;

  mode_type mode;   /* Run mode   CHECK_1PPS:  Look for 0xff[ff] each second
                                  CHECK_COUNT: Check consistency of data
				               from VSIC countdown mode */
  int verbose;      /* Prints whats happening */
  int quiet;        /* Less prints whats happening */
  int fail;         /* Don't keep going after a failure */
  int bits;         /* Number of bits/sample. Default 2*/
  int nchan;        /* Total number data channels. Default 4 */
  int bandwidth;    /* Recording bandwidth MHz */
  int maxbytes;     /* Maximum number of bytes to read in. Default all */
  //  int skip;         /* Number of bytes to skip at start of file */
  encodingtype encoding; /* VLBA or AT bit encoding (offset binary or sign/mag) */

  struct option options[] = {
    {"pps", 0, 0, 'p'},
    {"widepps", 0, 0, 'w'},
    {"count", 0, 0, 'x'},
    {"stats", 0, 0, 's'},
    {"bits", 1, 0, 'b'},
    {"channels", 1, 0, 'c'},
    {"search", 1, 0, 'S'},
    {"nbytes", 1, 0, 'n'},
    {"vlba", 0, 0, 'v'},
    {"fail", 0, 0, 'f'},
    {"verbose", 0, 0, 'V'},
    {"quiet", 0, 0, 'q'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  mode = MODE_NONE;
  quiet = 0;
  verbose = 0;
  fail = 0;
  stats = NULL;
  bits = -1;
  nchan = -1;
  //  bandwidth = -1;
  maxbytes = -1;
  encoding = AT;

  /* Keep compiler happy */
  psize = 0;
  pattern = NULL;
  poverlap = 0;
  last = 0;
  start = 0;
  expected = 0;
  wrap1 = 0;
  wrap2 = 0;
  pps = 0;
  widepps = 0;

  /* Read command line options */
  while (1) {
    opt = getopt_long_only(argc, argv, "svVqxvc:w:b:", options, NULL);
    if (opt==EOF) break;

    switch (opt) {
      
    case 'p':
      if (mode!=MODE_NONE) {
	fprintf(stderr, "Multiple modes requested aborting\n");
	return(1);
      }
      mode = CHECK_1PPS;
      break;

    case 'w':
      if (mode!=MODE_NONE) {
	fprintf(stderr, "Multiple modes requested aborting\n");
	return(1);
      }
      mode = CHECK_WIDEPPS;
      break;

    case 'x':
      if (mode!=MODE_NONE) {
	fprintf(stderr, "Multiple modes requested aborting\n");
	return(1);
      }
      mode = CHECK_COUNT;
      break;

    case 's':
      if (mode!=MODE_NONE) {
	fprintf(stderr, "Multiple modes requested aborting\n");
	return(1);
      }
      mode = CHECK_STATS;
      break;

    case 'b':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
 	fprintf(stderr, "Bad -bits option %s\n", optarg);
      else 
	bits = tmp;
      break;

    case 'c':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
 	fprintf(stderr, "Bad -channels option %s\n", optarg);
      else 
	nchan = tmp;
      break;

      //    case 'w':
      //      status = sscanf(optarg, "%d", &tmp);
      //      if (status!=1)
      // 	fprintf(stderr, "Bad -w (bandwidth) option %s\n", optarg);
      //      else 
      //	bandwidth = tmp;
      //      break;

    case 'S':
      if (mode!=MODE_NONE) {
	fprintf(stderr, "Multiple modes requested aborting\n");
	return(1);
      }
      if (strlen(optarg)>MAXSTR) {
	fprintf(stderr, "Search pattern too long (%d chars).\n", 
		(int)strlen(optarg));
	exit(1);
      }
      strcpy(searchstr, optarg);
      mode = SEARCH_PATTERN;
      break;

    case 'n':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
 	fprintf(stderr, "Bad -nbytes option %s\n", optarg);
      else 
	maxbytes = tmp;
      break;

    case 'V':
      verbose = 1;
      break;

    case 'q':
      quiet = 1;
      break;

    case 'f':
      fail = 1;
      break;

    case 'v':
      encoding = VLBA;
      break;

    case 'h':
      printf("Usage: vsib_checker [options]\n");
      printf("  -p/-pps          Check 1 PPS markers (default)\n");
      printf("  -x/-count        Check VSIC countdown mode\n");
      printf("  -s/-stats        Check sampler stats\n");
      printf("  -S/-search <hex> Search for bit pattern in data "
	     "(specified as hexnumber)\n");
      printf("  -b/-bits <b>     Total number bits/sample (8/16/32)\n");
      printf("  -n/-nbytes <n>   Maximum number of bytes to read per file\n");
      printf("  -v/-vlba         Calculate statistics for VLBA data\n");
      //      printf("  -w/bandwith <bw> Bandwidth of recorded data (2,4,8,16,32,64)"
      //	     "\n");
      printf("  -f/-fail         Stop running when problem in data found\n");
      printf("  -V/-verbose      Verbose\n");
      printf("  -q/-quiet        Less output\n");
      printf("  -h/-help         This list\n");
      return(1);
      break;

    case '?':
    default:
      break;
    }
  }
  
  if (mode==MODE_NONE) {
    mode = CHECK_1PPS;
    printf("Running in 1 PPS checking mode\n");
  }

  if (mode==SEARCH_PATTERN) {
    unsigned char *pptr;
    char *cptr, str[3];
    /* Convert the search string into a number */
    /* Assume the number is hex */
    psize = floor(strlen(searchstr)/2);
    pattern = malloc(psize);
    if (pattern==NULL) {
      sprintf(msg, "Trying to allocate %d byes", psize);
      perror(msg);
      return(1);
    }
    
    /* Assume no leading 0x or anything */
    cptr = searchstr;
    pptr = pattern;
    for (i=0; i<psize; i++) {
      strncpy(str, cptr, 2);
      if (!isxdigit(str[0]) || !isxdigit(str[1])) {
	fprintf(stderr, "Invalid digit %s in pattern\n", str);
	return(1);
      }
      status = sscanf(str, "%x", &tmp);
      if (status!=1) {
	fprintf(stderr, "Error converting %s into pattern byte\n", str);
	return(1);
      } else if (tmp>256) {
	fprintf(stderr, "Error converting %s into pattern "
		"- value too large\n", str);
	return(1);
      }
      *pptr = (unsigned char)tmp;
      
      cptr +=2;
      pptr++;
      
    }
  }

  header = newheader();

  for (nfile=optind; nfile<argc; nfile++) {
    /* Initialise stuff */

    file = open(argv[nfile], OPENOPTIONS);
    if (file==-1) {
      sprintf(msg, "Failed to open input file (%s)", argv[nfile]);
      perror(msg);
      return(1);
    }

    if (!quiet) printf("%s\n", argv[nfile]);

    resetheader(header);
    readheader(header, file, NULL);

    headsize = header->headersize;
    bandwidth = header->bandwidth;
    nchan = header->nchan;
    bits = header->numbits;

    /* Setup modes depending on bandwidth, number of channels etc */

    if (bandwidth==64)
      bytes = bits*nchan*4/8;
    else if (bandwidth==32)
      bytes = bits*2*nchan/8;
    else
      bytes = bits*nchan/8;

    if (mode==CHECK_WIDEPPS) bytes *=2;

    if (bytes==0)  bytes =  1;
    ppsskip = bandwidth*2*nchan*bits*1e6/8;

#if 0
    if (bandwidth==64) 
      ppsskip/= 4; /* 4 samples/byte */
    else if (bandwidth==32) 
      ppsskip/= 2; /* 2 samples/byte */
#endif

    if (mode==CHECK_COUNT || mode==CHECK_STATS || mode==SEARCH_PATTERN) {
      memsize = BLOCKSIZE;
    } else if (mode==CHECK_1PPS || mode==CHECK_WIDEPPS) {
      memsize = bytes;
    } else {
      fprintf(stderr, "Internal error. Unknown mode\n");
      return(1);
    }

    mem = malloc(memsize);
    if (mem==NULL) {
      sprintf(msg, "Trying to allocate %d byes", memsize);
      perror(msg);
      return(1);
    }

    if (mode==CHECK_STATS) {
      stats = calloc(256*sizeof(long)*bytes, 1);
      if (stats==NULL) {
	sprintf(msg, "Trying to allocate %d bytes", 256*(int)sizeof(long)*bytes);
	perror(msg);
	return(1);
      }

    }  else if (mode==CHECK_COUNT) {
      unsigned int decrement, temp1, temp2;
      unsigned char val8;
      unsigned short val16;
      int totalbits;

      switch (bandwidth) {
      case 64:
      case 16:
	decrement = 1;
	break;
      case 8:
	decrement = 2;
	break;
      case 4:
	decrement = 4;
	break;
      case 2:
	decrement = 8;
	break;
      case 32:
	fprintf(stderr, "Need to add support for 32 MHz bandwidth\n");
	exit(1);
	break;
      default:
	fprintf(stderr, "Unsupported bandwidth (%d MHz)\n", bandwidth);
	exit(1);
	break;
      }
    
      temp1 = 0;
      temp2 = 0;
      pps = 0;
      totalbits = bits*nchan;

      switch (totalbits) {
      case 8:
	/* The difference between samples */
	val8 = 0xFF;
	for (i=0; i<4; i++) {
	  temp1 |= val8<<i*8;
	  val8 -= decrement;
	}
	for (i=0; i<4; i++) {
	  temp2 |= val8<<i*8;
	  val8 -= decrement;
	}
	expected = temp1-temp2;

	/* The difference at a wrap */
	val8 = decrement-1;
	temp2 = 0x00;
	for (i=0; i<4; i++) {
	  temp2 |= val8<<(3-i)*8;
	  val8 += decrement;
	}

	wrap2 = temp1;
	wrap1 = temp2;

	pps = 0xff;
	break;
      
      case 16:
	val16 = 0xFFFF;
	/* The difference between samples */
	for (i=0; i<2; i++) {
	  temp1 |= val16<<i*16;
	  val16 -= decrement;
	}
	for (i=0; i<2; i++) {
	  temp2 |= val16<<i*16;
	  val16 -= decrement;
	}
	expected = temp1-temp2;
	//expected = 131074;      
	
	/* The difference at a wrap */
	val16 = decrement-1;
	temp2 = 0x00;
	for (i=0; i<2; i++) {
	  temp2 |= val16<<(1-i)*16;
	  val16 += decrement;
	}
	wrap2 = temp1;
	wrap1 = temp2;

	pps = 0xffff;
	break;
      case 32:
	expected = 1; 
	wrap1 = 0x0;
	wrap2 = 0x0;
	pps = 0xffffffff;
	break;
      case 2:
	expected = 0; 
	wrap1 = 0x0;
	wrap2 = 0x0;
	pps = 0xffffffff;
	break;
      default:
	expected = 0;
	wrap1 = 0x0;
	wrap2 = 0x0;
	pps = 0x0;
	printf("Do not support bit mode %d\n", totalbits);
	exit(1);
      }
    }

    //expected =524296  ;
    //wrap1 = 0;
    //wrap2 = 0;

    offset = 0;
    first = 1;
    secount = 1;
    while (1) {
      if (mode==CHECK_1PPS || mode==CHECK_WIDEPPS) {      
	if (!first) {
	  /* Skip to next 1 PPS marker */
	  off = lseek(file, ppsskip-bytes, SEEK_CUR); 
	  offset += ppsskip;
	  if (off==-1) {
	    sprintf(msg, "\nFailed seek to next second");
	    perror(msg);
	    close(file);
	    return(1);
	  } else if (off != offset+headsize) {
	    fprintf(stderr, "\nFailed seek to next second 2\n");
	    close(file);
	    return(1);
	  }
	} else {
	  first = 0;
	}
      }

      nread = read(file, mem, memsize);

      if (nread==0) {
	if (offset==0) {
	  fprintf(stderr, "No data read from %s\n", argv[nfile]);
	  free(mem);
	  close(file);
	  return(1);
	}
	break;
      } else if (nread==-1) {
	perror("Error reading file");
	close(file);
	free(mem);
	return(1);
      } else if (mode==CHECK_COUNT && nread%4) {
	/* Need to read in multiple of 4 bytes */
	off = lseek(file, -(nread%8), SEEK_CUR); 
	if (off==-1) {
	  perror(msg);
	  free(mem);
	  return(1);
	}
	nread -= nread%4;
      }

      if (verbose && (mode==CHECK_1PPS || mode==CHECK_WIDEPPS)) 
	  printf("Read second %d\n", secount);
      secount++;

      if (mode==CHECK_COUNT) {

	data = (unsigned int*)mem;
	for (i=0; i<nread/4; i++) {
	  if (first) {
	    first = 0;
	    last = *data;
	    data++;
	    continue;
	  }

	  //start = (unsigned short*) data;
	  diff = last - *data;

	  if (diff != expected) { /* Something is wrong */
	    int OK = 0;
	    if (last==wrap1 && *data==wrap2) OK = 1; // Counter wrapped
	    if (offset%ppsskip==0 &&  (last&pps)==pps) OK =1; /* Ignore 1 PPS */
	    if (offset%ppsskip==(ppsskip-4) &&  (*data&pps)==pps) OK =1;

	    if (!OK) {
	      printf("Bad values at offset %lld\n", (long long)offset);
	      printf("     0x%04x 0x%04x\n", last, *data);
	      //printf("     0x%04x 0x%04x 0x%04x 0x%04x\n", start[0], start[1], 
	      //     start[2], start[3]);
	      printf("   %d\n", diff);
	      if (fail) return(2);
	    }
	  }
	  last = *data;
	  data++;
	  offset += 4;
	}
      } else if (mode==CHECK_1PPS) {
	ppsOK = 1;
	switch (bits*nchan) {
	case 2:
	  if (((unsigned char)*mem&0xC) !=0xC) {
	    ppsOK = 0;
	    sprintf(msg, "0x%02x", (unsigned char)*mem);
	  }
	  break;

	case 4:
	  if (((unsigned char)*mem&0xF) !=0xF) {
	    ppsOK = 0;
	    sprintf(msg, "0x%02x", (unsigned char)*mem);
	  }
	  break;

	case 8:
	  if ((unsigned char)*mem!=0xFF) {
	    ppsOK = 0;
	    sprintf(msg, "0x%02x", (unsigned char)*mem);
	  }
	  break;

	case 16:
	  if (*(unsigned short*)mem != 0xFFFF) {
	    ppsOK = 0;
	    sprintf(msg, "0x%04x", (unsigned short)*mem);
	  }
	  break;

	case 32:
	  if (*(unsigned long*)mem != 0xFFFFFFFF) {
	    ppsOK = 0;
	    sprintf(msg, "0x%04lx", (unsigned long)*mem);
	  }
	  break;

	default:
	  fprintf(stderr, "Unsuported number of bits (%d)\n", bits);
	  exit(1);
	}

	if (!ppsOK) {
	  //printf("Missing PPS at offset 0x%lx (%s)\n", offset, msg);
	  printf("Missing PPS-- at offset %lld (%s)\n", (long long)offset, msg);
	  if (fail) return(2);
	}
      } else if (mode==CHECK_WIDEPPS) {
	ppsOK = 1;
	switch (bits*nchan) {

	case 2:
	  if (((unsigned char)*mem&0xF) == 0xF) {
	    ppsOK = 0;
	  }
	  break;

	case 4:
	  if ((unsigned char)*mem == 0xFF) {
	    ppsOK = 0;
	  }
	  break;

	case 8:
	  if (*(unsigned short*)mem == 0xFFFF) {
	    ppsOK = 0;
	  }
	  break;

	case 16:
	  if (*(unsigned long*)mem == 0xFFFFFFFF) {
	    ppsOK = 0;
	  }
	  break;

	case 32:
	  if (*(unsigned long*)mem == 0xFFFFFFFF && 
	      *(unsigned long*)(mem+4) == 0xFFFFFFFF) {
	    ppsOK = 0;
	  }
	  break;

	default:
	  fprintf(stderr, "Unsuported number of bits (%d)\n", bits);
	  exit(1);
	}

	if (!ppsOK && !widepps) {
	  widepps = 1;
	  printf("Wide PPS starts at sec %d\n", secount-2);
	} else if (widepps && ppsOK) {
	  printf("Wide PPS finishes at sec %d\n", secount-2);
	  widepps = 0;
	}
      } else if (mode==CHECK_STATS) {
	/* Check sampler statistics. If bits=8 just accumulate all
	   possible bit patterns. Otherwise we need to accumulate each
	   second byte seperately.  This will be interpreted at the
	   end of the file. Using a long we can accumulate a maximum
	   of 4 GB data (actually 10 with good statistics. 
	   The Huygens mode (8x16 MHz bands) and the BG3 mode (2 x 64 MHz)
	   needs to be done slightly differently.
 */
	if (offset+(unsigned int)nread>=UINT_MAX-1) { /* 4 samples/byte */
	  fprintf(stderr, "File too large (> 4 GB) for stat counters\n");
	  return(1);
	}
	ptr = mem;

	if (bytes==1) { /* Bits == 1,2,4,8 */
	  for (i=0; i<nread; i++) {
	    stats[*ptr]++;
	    ptr++;
	  }
	} else if (bytes==2) {
	  for (i=0; i<nread; i++) {
	    stats[*ptr +(i%2)*256]++;
	    ptr++;
	  }

	} else { /* Should add 32bit support */
	  fprintf(stderr, "Unsupported total number of bits\n");
	  return(1);
	}
	offset += nread;

      } else if (mode==SEARCH_PATTERN) {
	int n;
	unsigned char *mptr, *cptr;
	mptr = mem;
	n = nread;

	/* First check if there was a match crossing two buffer boundaries */
	if (poverlap>0) {
	  if (memcmp(mptr, pattern+poverlap, psize-poverlap)==0) {
	    printf("Found match at pos %lld (0x%llx)\n", 
		   (long long)offset-poverlap+headsize, 
		   (long long unsigned)offset-poverlap+headsize);
	    mptr += psize-poverlap;
	  } 
	  poverlap=0;
	}

	while (1) {
	  /* First search for the first character */
	  cptr = memchr(mptr, pattern[0], n);
	  if (cptr==NULL) {
	    break; /* Nothing to see here */
	  }
	  if (cptr+psize>mptr+n) { 
	    /* Found at end of buffer, need special handling */

	    if (memcmp(cptr, pattern, mptr+n-cptr)==0) {  
	      /* A match at the end */
	      poverlap = mptr+n-cptr;
	      break;
	    }
	  } else {
	    if (memcmp(cptr, pattern, psize)==0) { /* They are the same */
	      printf("Found match at pos %lld (%llx)!!\n", 
		     (long long)(cptr-mem)+offset+headsize, 
		     (unsigned long long)(cptr-mem)+offset+headsize);
	      cptr+= psize-1;
	    }
	  }
	  
	  mptr = cptr+1;
	  n = nread - (mptr-mem);
	}

	offset += nread;
      } else {
	fprintf(stderr, "Internal error 2\n");
	return(1);
      }
      if (maxbytes>0 && offset>maxbytes) break;
    }
    status = close(file);

    if (mode==CHECK_STATS) {
      unsigned long avstats[16][4], totalcounts;
      int decode[4];
      char bitstr[4][3] = {"00", "01", "10", "11"};
      
      if (encoding==AT) {
	decode[0] = 3;
	decode[1] = 1;
	decode[2] = 0;
	decode[3] = 2;
      } else { /* Must be VLBA */
	decode[0] = 3;
	decode[1] = 2;
	decode[2] = 1;
	decode[3] = 0;
      }

      for (i=0; i<16; i++) {
	for (j=0; j<4; j++) {
	  avstats[i][j] = 0;
	}
      }
      totalcounts = 0;

      /* Need to report. Maybe change this to every second */

      if (bits==2) {
	printf("         high-  low-   low+   high+\n");
	printf("      ");
	for (j=0; j<4; j++) {
	  printf("     %s", bitstr[decode[j]]);
	}
	printf("\n");

	if (bytes==1) {
	  for (i=0; i<256; i++) { /* Go through each byte distribution */
	    for (j=0; j<4; j++) { /* Do each 2 bit sample seperately */
	      avstats[j][i>>(j*2)&0x3] += stats[i];
	    }
	    totalcounts += stats[i];
	  }

	  /* Sum channels for multisample/byte */

	  if (bandwidth==64 || (bandwidth==32 && nchan==1)) {
	    /* 4 bytes/sample */

	    for (j=0; j<4; j++) {
	      avstats[0][j] += avstats[1][j];
	      avstats[0][j] += avstats[2][j];
	      avstats[0][j] += avstats[3][j];
	    }
	    totalcounts *= 4;

	  } else if (bandwidth==32) { /* nchan must equal 2 */
	    for (j=0; j<4; j++) {
	      avstats[0][j] += avstats[1][j];
	      avstats[1][j] = avstats[2][j];
	      avstats[1][j] += avstats[3][j];
	    }
	    totalcounts *= 2;

	  } else if (nchan==1) {
	    for (i=1; i<4; i++) {
	      for (j=0; j<4; j++) {
		avstats[0][j] += avstats[i][j];
	      }
	    }
	    totalcounts *= 4;

	  } else if (nchan==2) {
	    for (i=0; i<2; i++) {
	      for (j=0; j<4; j++) {
		avstats[i][j] += avstats[i+2][j];
	      }
	    }
	    totalcounts *= 2;
	  }

	} else if (bytes==2) {
	  for (i=0; i<256; i++) { /* Go through each byte distribution */
	    for (j=0; j<4; j++) { /* Do each 2 bit sample separately */
	      avstats[j][i>>(j*2)&0x3] += stats[i];
	      avstats[j+4][i>>(j*2)&0x3] += stats[256+i];
	    }
	    totalcounts += stats[i] + stats[256+i];
	  }
	  totalcounts /= 2; /* We have counted twice */

	  if (bandwidth==64) {
	    /* 4 bytes/sample */

	    for (j=0; j<4; j++) {
	      avstats[0][j] += avstats[1][j];
	      avstats[0][j] += avstats[2][j];
	      avstats[0][j] += avstats[3][j];
	    }
	    for (j=0; j<4; j++) {
	      avstats[1][j] = avstats[4][j];
	      avstats[1][j] += avstats[5][j];
	      avstats[1][j] += avstats[6][j];
	      avstats[1][j] += avstats[7][j];
	    }
	    totalcounts *= 4;
	  } else if (bandwidth==32) {
	    for (i=0; i<4; i++) {
	      for (j=0; j<4; j++) {
		avstats[i][j] = avstats[i*2][j];
		avstats[i][j] += avstats[i*2+1][j];
	      }
	    }
	    totalcounts *= 2;
	  }
	}

	for (i=0; i<nchan; i++) {
	  printf("Chan %d:", i);
	  for (j=0; j<4; j++) {
	    printf(" %6.2f", avstats[i][decode[j]]/(float)totalcounts*100);
	  }
	  printf("\n");
	}
      } else if (bits==1) {
	printf("         -    +\n");

	if (bytes==1) {
	  for (i=0; i<256; i++) { /* Go through each byte distribution */
	    for (j=0; j<8; j++) { /* Do each 1 bit sample seperately */
	      avstats[j][(i>>j)&0x1] += stats[i];
	    }
	    totalcounts += stats[i];
	  }

	  /* Sum channels for multisample/byte */

	  if (nchan==1) {
	    for (i=1; i<8; i++) {
	      avstats[0][0] += avstats[i][0];
	      avstats[0][1] += avstats[i][1];
	    }
	    totalcounts *= 8;

	  } else if (nchan==2) {
	    for (j=1; j<4; j++) {
	      for (i=0; i<2; i++) {
		avstats[0][i] += avstats[j*2][i];
		avstats[1][i] += avstats[j*2+1][i];
	      }
	    }
	    totalcounts *= 4;

	  } else if (nchan==4) {
	    for (i=0; i<4; i++) {
	      avstats[i][0] += avstats[i+4][0];
	      avstats[i][1] += avstats[i+4][1];
	    }
	    totalcounts *= 2;
	  }
	} else if (bytes==2) {
	  fprintf(stderr, "Do not support this mode\n");
	  exit(1);
	}

	for (i=0; i<nchan; i++) {
	  printf("Chan %d:", i);
	  for (j=0; j<2; j++) {
	    printf(" %6.2f", avstats[i][j]/(float)totalcounts*100);
	  }
	  printf("\n");
	}
	
      } else {
	


	fprintf(stderr, "Do not support %d bit modes\n", bits);
	exit(1);
      }
      printf("Total=%ld\n", totalcounts);
   

      free(mem);
      free(stats);
    }
  }

  destroyheader(header);
  if (mode==SEARCH_PATTERN)
    free(pattern);
  
  return(0);
}

  
