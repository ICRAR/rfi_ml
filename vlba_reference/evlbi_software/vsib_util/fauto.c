
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
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <cpgplot.h>
#include <complex.h>
#include <math.h>
#include <fftw3.h>
#include <sys/ipc.h>  /* for shared memory */
#include <sys/shm.h>
#include "../vsib/sh.h"

#include "../vsib/vheader.h"

#define MALLOC fftwf_malloc

#define MIN(x,y)   ((x) < (y) ? (x) : (y))

#define MAX(x,y)   ((x) > (y) ? (x) : (y))

#define MAXCHAN 8

double read_head(int file, struct tm *tm);


void doplot(int nchan, int chans[], int npoint, int bchan, int echan, 
	    int naver, int swapped, int nx, int domax, int doavg, 
            int dolog, int dooverplot, int crosspol, float ymax,  float ymin,
	    float xvals[], double *spectrum[MAXCHAN], fftwf_complex *cross[MAXCHAN/2],
	    float plotspec[], char outfile[], double secperfft, double mjd);
double tm2mjd(struct tm date);

#define BLOCKSIZE 4*1024*1024

#define RATE 32
#define MAXSTR 1024

double tim(void) {
  struct timeval tv;
  double t;

  gettimeofday(&tv, NULL);
  t = (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;

  return t;
}

int main (int argc, char * const argv[]) {
  int nfile, i, j, k, l, file, status, nread, opt, memsize, tmp, thisread, dobreak;
  int naver, readshort, nfft, nx, ny, swapped, nplot, bytesperfft, bytesperint, first;
  register int samp;
  char msg[MAXSTR+1], *filename, *outfile, *cptr;
  unsigned char  *mem;
  unsigned short *mem16;
  char *sample;
  int16_t *sample16;
  //uint16_t *usamp16;
  off_t off, offset, offset2, filesize;
  float *in[MAXCHAN], lookup[4], *plotspec, *xvals, ftmp;
  fftwf_complex *out[MAXCHAN], *cross[MAXCHAN/2];
  double *spectrum[MAXCHAN], t0, t1, tt, secperfft, secpersamp, mjd, startmjd, tA, tB;
  long long lcount[4], tcount;
  fftwf_plan plans[MAXCHAN]; /* Do up to MAXCHAN subbands at the same time */

  int verbose;       /* Prints whats happening */
  int online;        /* Run continously on live data */
  int pause;         /* Wait between subplots */
  int crosspol;      /* Compute crosspols */
  int normalise;     /* Subtract average from bits > 2 */
  int faststart;     /* Fast fftw planning */
  int dump;          /* Write output spectrum */
  int domax;         /* Just print maximum value (used for blanking tests) */
  int doavg;         /* Just print average 80% of band */
  int dolog;         /* Take log10 of data before plotting */
  int dooverplot;    /* Don't erase between plots */
  int px14;          /* Data for p14400  - Assumed 64 MHz */
  int bits;          /* Number of bits/sample */
  int nchan;         /* Total number of channels */
  int bandwidth;     /* Observing bandwidth (different byte format for 32 and 
			64 MHz data*/
  int nskip;         /* Number of bytes to skip at start of file */
  int npoint;        /* Number of spectral channels */
  int nint;          /* Integration time, in number of ffts */
  float tint;        /* Integration time in seconds */
  int chans[MAXCHAN]; /* Channels to correlate */
  int bchan;         /*  Frequency offset of first spectral point to plot */
  float bechan;        /* Frequency offset of last spectral point to plot */
  float bbchan;        /* First spectral point to plot */
  int echan;         /* Last spectral point to plot */
  int docommand;     /* Command string passed? */
  float ymax;        /* Maximum yvalue of plot */
  float ymin;        /* Minimum yvalue of plot */
  encodingtype encoding; /* VLBA or AT bit encoding (offset binary or sign/mag) */
  char pgdev[MAXSTR+1] = "/xs"; /* Pgplot device to use */
  char command[MAXSTR+1] = ""; /* Pgplot device to use */

  int chan1, chan2, actualbits;
  vhead *header=0;
  struct stat filestat;
  struct tm time;

  /* Shared memory unique identifier (key) and the returned reference id. */
  key_t shKey;
  int shId = -1;
  ptSh sh;

  struct option options[] = {
    {"vlba", 0, 0, 'v'},
    {"px14", 0, 0, 'x'},
    {"npoint", 1, 0, 'n'},
    {"chan", 1, 0, 'C'},
    {"nint", 1, 0, 'N'},
    {"tint", 1, 0, 't'},
    {"skip", 1, 0, 's'},
    {"if1", 1, 0, '1'},
    {"if2", 1, 0, '2'},
    {"sp1", 1, 0, 'q'},
    {"sp2", 1, 0, 'r'},
    {"bw1", 1, 0, 'Q'},
    {"bw2", 1, 0, 'R'},
    {"ymax", 1, 0, 'y'},
    {"ymin", 1, 0, 'Y'},
    {"pause", 0, 0, 'P'},
    {"crosspol", 0, 0, 'p'},
    {"normalise", 0, 0, 'b'},
    {"faststart", 0, 0, 'f'},
    {"dump", 0, 0, 'D'},
    {"max", 0, 0, 'm'},
    {"log", 0, 0, 'l'},
    {"overplot", 0, 0, 'O'},
    {"average", 0, 0, 'a'},
    {"command", 1, 0, 'Z'},
    {"device", 1, 0, 'd'},
    {"pgdev", 1, 0, 'd'},
    {"verbose", 0, 0, 'V'},
    {"online", 0, 0, 'o'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  verbose = 0;
  online = 0;
  pause = 0;
  crosspol = 0;
  dump = 0;
  normalise = 0;
  domax = 0;
  doavg = 0;
  dolog = 0;
  px14 = 0;
  dooverplot = 0;
  nskip = 0;
  docommand = 0;
  npoint = 128;
  encoding = AT;
  chan1 = -1;
  chan2 = -1;
  nint = -1;
  tint = -1;
  for (k=0;k<MAXCHAN;k++) chans[k]=0;
  bchan = -1;
  echan = -1;
  bbchan = -1;
  bechan = -1;
  ymax = 0.0;
  ymin = 0.0;
  faststart = 0;
  sh = NULL;
  filename = NULL;

  /* Avoid compiler complaints */
  memsize = 0;
  nx = 0;
  swapped = 0;
  bytesperfft = 0;
  bytesperint = 0;
  secperfft = 0;
  secpersamp = 0;
  mem = 0;
  bits = 0;
  nchan = 0;
  bandwidth = 0.0;
  mjd = 0.0;
  startmjd = 0.0;

  nfft = 0;
  /* Read command line options */
  while (1) {
    opt = getopt_long_only(argc, argv, "d:N:s:y:n:c:C:Vpvo1:2:", 
			   options, NULL);
    if (opt==EOF) break;

    switch (opt) {
      
    case 'C':
    case 'c':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
 	fprintf(stderr, "Bad -chan option %s\n", optarg);
      else {
	if (tmp>MAXCHAN) {
	  fprintf(stderr, "Chan %s too large\n", optarg);
	} else if (tmp<=0) {
	  fprintf(stderr, "Chan %s too small\n", optarg);
	} else {
	  chans[tmp-1] = 1;
	}
      }
      break;

    case '1':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
 	fprintf(stderr, "Bad -1chan option %s\n", optarg);
      else 
	chan1 = tmp-1;
      break;

    case '2':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
 	fprintf(stderr, "Bad -2chan option %s\n", optarg);
      else 
	chan2 = tmp;
      break;

    case 'q':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
 	fprintf(stderr, "Bad -sp1 option %s\n", optarg);
      else 
	bchan = tmp;
      break;

    case 'r':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
 	fprintf(stderr, "Bad -sp2 option %s\n", optarg);
      else 
	echan = tmp;
      break;

    case 'Q':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1)
 	fprintf(stderr, "Bad -bw1 option %s\n", optarg);
      else 
	bbchan = ftmp;
      break;

    case 'R':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1)
 	fprintf(stderr, "Bad -bw2 option %s\n", optarg);
      else 
	bechan = ftmp;
      break;

    case 'y':
      printf("Got ymax = %s\n", optarg);
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1)
 	fprintf(stderr, "Bad -ymax option %s\n", optarg);
      else 
	ymax = ftmp;
      break;

    case 'Y':
      printf("Got ymin = %s\n", optarg);
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1)
 	fprintf(stderr, "Bad -ymin option %s\n", optarg);
      else 
	ymin = ftmp;
      break;

    case 's':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
 	fprintf(stderr, "Bad -skip option %s\n", optarg);
      else 
	nskip = tmp;
      break;

    case 'n':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
 	fprintf(stderr, "Bad -npoint option %s\n", optarg);
      else 
	npoint = tmp;
      break;

    case 'N':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
 	fprintf(stderr, "Bad -nint option %s\n", optarg);
      else 
	nint = tmp;
      break;

    case 't':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1)
 	fprintf(stderr, "Bad -tint option %s\n", optarg);
      else 
	tint = ftmp;
      break;

    case 'd':
      if (strlen(optarg)>MAXSTR) {
	fprintf(stderr, "PGDEV option %s too long\n", optarg);
	return 0;
      }
      strcpy(pgdev, optarg);
      break;

    case 'Z':
      if (strlen(optarg)>MAXSTR) {
	fprintf(stderr, " -command option %s too long\n", optarg);
	return 0;
      }
      strcpy(command, optarg);
      docommand = 1;
      break;

    case 'V':
      verbose = 1;
      break;

    case 'o':
      online = 1;
      break;

    case 'P':
      pause = 1;
      break;

    case 'p':
      crosspol = 1;
      break;

    case 'b':
      normalise = 1;
      break;

    case 'f':
      faststart = 1;
      break;

    case 'D':
      dump = 1;
      break;

    case 'm':
      domax = 1;
      break;

    case 'l':
      dolog = 1;
      break;

    case 'x':
      px14 = 1;
      break;

    case 'O':
      dooverplot = 1;
      break;

    case 'a':
      doavg = 1;
      break;

    case 'v':
      encoding = VLBA;
      break;

    case 'h':
      printf("Usage: fauto [options]\n");
      printf("  -C/-chan <n>        Channel to correlate (can specify "
	     "multiple times)\n");
      printf("  -n/-npoint <n>      # spectral channels\n");
      printf("  -v/-vlba            Calculate statistics for VLBA data\n");
      printf("  -N/-init <val>      Number of ffts to average per "
	     "integration\n");
      printf("  -device <pgdev>     Pgplot device to plot to\n");
      printf("  -if1/-1 <n>         1st channel to correlate in range\n");
      printf("  -if2/-2 <n>         Last channel to correlate in range\n");
      printf("  -sp1 <n>            First spectral point to plot\n");
      printf("  -sp2 <n>            Last spectral point to plot\n");
      printf("  -s/-skip <n>        Skip this many bytes at start of file\n");
      printf("  -p/-pause           Pause between fft subintegrations\n");
      printf("  -D/-dump            Write ascii copy of spectrum\n");
      printf("  -V/-verbose         Verbose\n");
      printf("  -h/-help            This list\n");
      return(1);
      break;

    case '?':
    default:
      break;
    }
  }

  if (docommand && ! dump) {
    fprintf(stderr, "Can only run command when writing spectra!\n");
    docommand = 0;
  }

  if (tint>0 && nint>0) {
    fprintf(stderr, "Cannot set both -tint and -nint!  Ignoring tint\n");
    tint = -1;
  }


  /* Initiate plotting arrays */

  xvals = MALLOC(sizeof(float)*npoint);
  for (j=0; j<npoint; j++) {
    xvals[j] = j+0.5;
  }
  plotspec = MALLOC(sizeof(float)*npoint);

  /* Setup the lookup table */
  if (encoding==VLBA) {
    lookup[0] = +3.0;
    lookup[1] = -1.0;
    lookup[2] = +1.0;
    lookup[3] = -3.0;
  } else { /* AT */
    lookup[0] = +1.0;
    lookup[1] = -1.0;
    lookup[2] = +3.0;
    lookup[3] = -3.0;
  }

  if (online) {
  /* Create and initialize shared memory. */
    shKey = fourCharLong('v','s','i','b');

    shId = shmget(shKey, 0, 0);
    if (shId==-1) {
      perror("shmget failed"); 
      exit(1);
    }
      
    sh = (ptSh)shmat(shId, NULL, 0);
    if (sh == (void *)-1 ) {
      perror("shmat failed");
      exit(1);
    }

    filename = (char *)malloc(SH_MAXSTR); 
    if (filename==NULL) {
      perror("Allocating memory");
      exit(1);
    }
    strcpy(filename,"");
  }

  if (dump) {
    outfile = (char *)malloc(MAXSTR+1);
    if (outfile==NULL) {
      perror("Allocating memory");
      exit(1);
    }
  } else {
    outfile = NULL;
  }

  tt = 0; /* Total time */
  naver = 0;
  first = 1;

  nfile = optind;
  if (!px14) header = newheader();

  tA = tim();
  while (1) { // Loop for online mode
    while (1) { // Loop over (and accumulate) input files

      if (online) {
	if (strcmp(filename,"")==0 || strcmp(sh->currentfile,filename)!=0) {
	  strcpy(filename, sh->currentfile);
	} else {
	  sleep(1);
	  continue;
	}
	
      } else {
	if (nfile >= argc) break;
	filename = argv[nfile];
	nfile++;
      }

      file = open(filename, OPENOPTIONS);
      if (file==-1) {
	sprintf(msg, "Failed to open input file (%s)", filename);
	perror(msg);
	//fftwf_free(mem);
	return(1);
      }
      printf("%s\n", filename);

      if (!px14) {
	/* Read the header */
	resetheader(header);
	readheader(header, file, NULL);

	if (first) {
	  // Assume all files the same, for now
	  nchan = header->nchan;
	  bits = header->numbits;
	  bandwidth = header->bandwidth;
	}

	gettime(header, &time);
	mjd = startmjd = tm2mjd(time);
	

      } else {
	nchan = 2;
	bits = 16;
	bandwidth = 64;
      }

      if (bits==10) 
	actualbits = 16;
      else
	actualbits = bits;
      

      if (first) {
	/* Check too high channel hasn't been selected */
	for (k=nchan; k<MAXCHAN; k++) {
	  if (chans[k]) {
	    fprintf(stderr, "Channel %d is too large for this mode. Ignoring\n", k+1);
	  chans[k] = 0;
	  }
	}
	
	if (chan1>0 || chan2>0) {
	  if (chan1<=0) 
	    chan1=1;
	  else if (chan1>nchan) {
	    fprintf(stderr, "Channel %d too large for current mode. Setting to %d\n",
		    chan1, nchan);
	    chan1=nchan;
	  }
	  
	  if (chan2<=0) 
	    chan2=nchan;
	  else if (chan2>nchan) {
	    fprintf(stderr, "Channel %d too large for current mode. Setting to %d\n",
		    chan2, nchan);
	    chan2=nchan;
	  }
	  if (chan2<chan1) {
	    tmp = chan1;
	    chan1 = chan2;
	    chan2 = tmp;
	  }
	  for (k=chan1-1;k<chan2;k++) {
	    chans[k] = 1;
	  }
	}
	
	/* Have any channels been selected? */
	nplot = 0;
	for (k=0; k<nchan; k++) {
	  if (chans[k]) {
	    nplot++;
	  }
	}
	if (nplot==0) {
	  fprintf(stderr, "Plotting all channels\n");
	  
	  for (k=0; k<nchan; k++) {
	    chans[k] = 1;
	  }
	  nplot=nchan;
	}

	if (bbchan>0 && bchan<0) {
	  bchan = bbchan/bandwidth * npoint;
	}
	if (bechan>0 && echan<0) {
	  echan = bechan/bandwidth * npoint;
	}

	if (bchan<0) bchan = 0;
	if (echan<0) echan = npoint-1;
	if (bchan > npoint-1) bchan = npoint-1;
	if (echan > npoint-1) echan = npoint-1;
	if (bchan==echan) {
	  fprintf(stderr, "Warning: Odd channel range. Resetting\n");
	  bchan = 0;
	  echan = npoint-1;
	}
  
	if (bchan>echan) {
	  swapped = bchan;
	  bchan = echan;
	  echan = swapped;
	  swapped = 1;
	  fprintf(stderr, "Swapping frequency axis\n");
	} 
	echan++;
	
	/* Bytes per integration */
	bytesperfft = npoint*2*actualbits*nchan/8;

	secpersamp = 1/(bandwidth*2*1e6);
	secperfft = (2*npoint)*secpersamp;
	if (tint>0) {
	  // Closest # ffts per intergration 
	  nint = rint(tint/secperfft);
	  printf("Using %d fft per integration\n", nint);
	}
	bytesperint = nint*bytesperfft;

	printf("Bytes per fft = %d\n", bytesperfft);
	
	if (nint>0) {
	  memsize = bytesperint;
	} else {
	  memsize = floor(BLOCKSIZE/bytesperfft)*bytesperfft+0.1;
	}

	mem = MALLOC(memsize);
	if (mem==NULL) {
	  sprintf(msg, "Trying to allocate %d byes", memsize);
	  perror(msg);
	  return(1);
	}

	/* Initiate fftw */
	
	for (k=0; k<nchan; k++) {
	  if (!chans[k]) continue;
	  in[k] = MALLOC(sizeof(float) * npoint*2);
	  if (in[k]==NULL) {
	    perror("Allocating memory");
	    exit(1);
	  }
	  out[k] = MALLOC(sizeof(fftwf_complex) *npoint);
	  if (out[k]==NULL) {
	    perror("Allocating memory");
	    exit(1);
	  }
	  spectrum[k] = MALLOC(sizeof(double)*npoint);
	  if (spectrum[k]==NULL) {
	    perror("Allocating memory");
	    exit(1);
	  }
	  for (j=0; j<npoint; j++) {
	    spectrum[k][j] = 0;
	  }
	  
	  if (faststart)
	    plans[k] = fftwf_plan_dft_r2c_1d(npoint*2, in[k], out[k], FFTW_ESTIMATE);
	  else 
	    plans[k] = fftwf_plan_dft_r2c_1d(npoint*2, in[k], out[k], 0);
	  //plans[k] = fftwf_plan_dft_r2c_1d(npoint*2, in[k], out[k], FFTW_MEASURE);
	}

	if (crosspol) {
	  for (k=0; k<nchan/2; k++) {
	    cross[k] = MALLOC(sizeof(fftwf_complex)*npoint);
	    if (cross[k]==NULL) {
	      perror("Allocating memory");
	      exit(1);
	    }
	    for (j=0; j<npoint; j++) {
	      cross[k][j] = 0;
	    }
	  }
	}

	/* How many plots to do */;
	int totalplot = nplot;
	if (crosspol) totalplot += nplot/2;
	ny = ceil(sqrt(totalplot));
	nx = (totalplot+ny-1)/ny;
	
	if (!domax && !doavg) {
	  if (cpgbeg(0,pgdev,nx,ny)!=1) {
	    fprintf(stderr, "Error calling PGBEGIN");
	    return(1);
	  }
	  cpgask(0);
	}
      }
    
      for (i=0; i<4; i++) {
	lcount[i] = 0;
      }

      offset = 0;
      readshort = 0;

      if (nskip>0) {
	offset = lseek(file, nskip, SEEK_CUR);
	if (offset==-1) {
	  sprintf(msg, "Trying to skip %d byes", nskip);
	  perror(msg);
	  //fftwf_free(mem);
	  return(1);
	}

	mjd += nskip*secpersamp/(nchan*actualbits)*8/(60*60*24);

      }


      while (1) { 
	if (naver==0 && online && nint>0) { // Seek to appropriate place in file - start of current integration
	  status =  fstat(file, &filestat);
	  if (status != 0) {
	    sprintf(msg, "Trying to stat %s", filename);
	    perror(msg);
	    return(1);
	  }
	  filesize = filestat.st_size;
	  offset = lseek(file, 0, SEEK_CUR);
	  if (offset==-1) {
	    sprintf(msg, "Trying to get current file position");
	    perror(msg);
	    return(1);
	  }

	  if (filesize-offset > bytesperint) {
	    offset = ((filesize-header->headersize)/bytesperint)*bytesperint + header->headersize;
	    offset2 = lseek(file, offset, SEEK_SET);
	    if (offset2==-1) {
	      sprintf(msg, "Trying to seek to start of integration at %lu bytes", (long)offset2);
	      perror(msg);
	      return(1);
	    } else if (offset2 != offset) {
	      fprintf(stderr, "Failed to seek to start of integration = got %lu not %lu\n", (unsigned long)offset2, (unsigned long)offset);
	    }
	    mjd = startmjd + (offset-header->headersize)*secpersamp/(nchan*actualbits)*8/(60*60*24);
	  }
	}

	nread = 0;
	dobreak = 0;
	while (nread < memsize) { 
	  // Try and read a little more
	  
	  thisread = read(file, mem+nread, memsize-nread);
	  if (thisread==0) {  // EOF
	    if (online) {
	      if (strcmp(sh->currentfile,filename)!=0) { // new file
		dobreak = 1;
		break;
	      } else {
		// Sleep a bit and try again
		usleep(50000); // 50msec
		continue;
	      }
	    } else {
	      if (offset<=nskip) 
		fprintf(stderr, "No data read from %s\n", filename);
	      dobreak = 1;
	      break;
	    }
	  } else if (nread==-1) {
	    perror("Error reading file");
	    close(file);
	    return(1);
	  }
	  nread += thisread;
	}
	if (dobreak) break;

	if (nread%bytesperfft != 0) { /* Need to read multiple of lags */
	  /* Data may be pack with multiple samples/byte or mutiple 
	     bytes/sample */
	  int shift;
	  shift = nread % bytesperfft;
	  
	  off = lseek(file, -shift, SEEK_CUR); 
	  if (off==-1) {
	    perror(msg);
	    //fftwf_free(mem);
	    return(1);
	  }
	  nread -= shift;
	}

	offset += nread;
      
	mem16 = (unsigned short*)mem;

	/* The following loops decode the data into a set of float arrays */

	t0 = tim();
	/* Copy data into "in" array, fft then accumulate */
	for (i=0; i<nread/bytesperfft; i++) {

	  if (bits==2) {
	    if (bandwidth==64) { /* 64 MHz needs to be handle separately */
	      if (nchan==2) {
		for (j=0; j<npoint/2; j++) {
		  for (k=0; k<nchan; k++) {
		    if (!chans[k]) continue;
		    samp = mem[i*bytesperfft+j*2+k];
		    for (l=0; l<4; l++) {
		      in[k][j*4+l] = lookup[(samp>>(6-l*2))&0x3];
		      lcount[(samp>>(l*2))&0x3]++;
		    }
		  }
		}
	      } else if (nchan==1) {
		for (j=0; j<npoint/2; j++) {
		  samp = mem[i*bytesperfft+j];
		  for (l=0; l<4; l++) {
		    in[0][j*4+l] = lookup[(samp>>(6-l*2))&0x3];
		    lcount[(samp>>(l*2))&0x3]++;
		  }
		}
	      } else {
		fprintf(stderr, "64 MHz %d channels not supported\n", nchan);
		exit(1);
	      }
	    
	    } else if (bandwidth==32) {
	      if (nchan==1) {
		for (j=0; j<npoint/2; j++) {
		  samp = mem[i*bytesperfft+j];
		  for (l=0; l<4; l++) {
		    in[0][j*4+l] = lookup[(samp>>(l*2))&0x3];
		    lcount[(samp>>(l*2))&0x3]++;
		  }
		}
	      } else if (nchan==2) {
		for (j=0; j<npoint; j++) {
		  samp = mem[i*bytesperfft+j];
		  for (k=0; k<nchan; k++) {
		    if (!chans[k]) continue;
		    in[k][j*2] = lookup[(samp>>k*4)&0x3];
		    in[k][j*2+1] = lookup[(samp>>(k*4+2))&0x3];
		    lcount[(samp>>(k*4))&0x3]++;
		    lcount[(samp>>(k*4+2))&0x3]++;
		  }
		}
	      } else if (nchan==4) {
		for (j=0; j<npoint; j++) {
		  samp = mem16[i*npoint+j];
		  for (k=0; k<4; k++) {
		    if (!chans[k]) continue;
		    in[k][j*2] = lookup[(samp>>(k*4))&0x3];
		    in[k][j*2+1] = lookup[(samp>>(k*4+2))&0x3];
		    lcount[(samp>>(k*4))&0x3]++;
		    lcount[(samp>>(k*4+2))&0x3]++;
		  }
		}
	      } else {
		fprintf(stderr, "32 MHz %d channels not supported\n", nchan);
		exit(1);
	      }

	    } else { // Normal modes. Bandwidth unimportant
	      if (nchan==4) {
		for (j=0; j<npoint*2; j++) {
		  samp = mem[i*npoint*2+j];
		  for (k=0; k<nchan; k++) {
		    if (!chans[k]) continue;
		    in[k][j] = lookup[(samp>>(k*2))&0x3];
		    lcount[(samp>>(k*2))&0x3]++;
		  }
		}
		
	      } else if (nchan==8) {
		for (j=0; j<npoint*2; j++) {
		  samp = mem16[i*npoint*2+j];
		  for (k=0; k<nchan; k++) {
		    if (!chans[k]) continue;
		    in[k][j] = lookup[(samp>>(k*2))&0x3];
		    lcount[(samp>>(k*2))&0x3]++;
		  }
		}
	      } else if (nchan==2) {
		for (j=0; j<npoint; j++) {
		  samp = mem[i*bytesperfft+j];
		  for (k=0; k<nchan; k++) {
		    if (!chans[k]) continue;
		    in[k][j*2] = lookup[(samp>>(k*2))&0x3];
		    in[k][j*2+1] = lookup[(samp>>(k*2+4))&0x3];
		    lcount[(samp>>(k*2))&0x3]++;
		    lcount[(samp>>(k*2+4))&0x3]++;
		  }
		}
	      } else if (nchan==1) {
		for (j=0; j<npoint/2; j++) {
		  samp = mem[i*bytesperfft+j];
		  
		  in[0][j*4] = lookup[samp&0x3];
		  in[0][j*4+1] = lookup[(samp>>2)&0x3];
		  in[0][j*4+2] = lookup[(samp>>4)&0x3];
		  in[0][j*4+3] = lookup[(samp>>6)&0x3];
		  
		  lcount[samp&0x3]++;
		  lcount[(samp>>2)&0x3]++;
		  lcount[(samp>>4)&0x3]++;
		  lcount[(samp>>6)&0x3]++;
		}
	      } else {
	      fprintf(stderr, "Unsupported number of channels %d\n", nchan);
	      exit(1);
	      }
	    }
	    
	  } else if (bits==8) {
	    sample =  (char*)&mem[i*bytesperfft];
	    sample--;
	    for (j=0; j<npoint*2; j++) {
	      for (k=0; k<nchan; k++) {
		sample++;
		if (!chans[k]) continue;
		in[k][j] = *sample;
	      }
	    }
	  } else if (bits==10) {
 	    sample16 =  (int16_t*)&mem16[i*bytesperfft/2];
	    sample16--;
	    for (j=0; j<npoint*2; j++) {
	      for (k=0; k<nchan; k++) {
		sample16++;
		if (*sample16&0x200) // 10th bit set
		  *sample16 |= 0xFC00; // Force upper 6 bits high;
	      else
		*sample16 &= 0x03FF; // Force upper 6 bits low;
		
		if (!chans[k]) continue;
		in[k][j] = *sample16;
	      }
	    }
	  } else if (bits==16) {
#if 0
	    usamp16 =  (uint16_t*)&mem16[i*bytesperfft/2];
	    usamp16--;
	    for (j=0; j<npoint*2; j++) {
	      for (k=0; k<nchan; k++) {
		usamp16++;
		if (!chans[k]) continue;
		in[k][j] = (float)(*usamp16)-INT16_MAX;
	      }
	    }
#else
	    sample16 =  (int16_t*)&mem16[i*bytesperfft/2];
	    sample16--;
	    for (j=0; j<npoint*2; j++) {
	      for (k=0; k<nchan; k++) {
		sample16++;
		if (!chans[k]) continue;
		in[k][j] = *sample16;
	      }
	    }
#endif
	  } else {
	    fprintf(stderr, "Unsupported number of bits %d\n", bits);
	    exit(1);
	  }

	  if (normalise && (bits>2)) {
	    float sum;
	    for (k=0; k<nchan; k++) {
	      if (!chans[k]) continue;
	      sum = 0;
	      for (j=0; j<npoint*2; j++) sum += in[k][j];
	      sum /= npoint*2;
	      for (j=0; j<npoint*2; j++) in[k][j] -= sum;
	    }
	  }
	  
	  for (k=0; k<nchan; k++) {
	    if (!chans[k]) continue;
	    
	    fftwf_execute(plans[k]);
	    nfft++;
	    for (j=0; j<npoint; j++) {
	      spectrum[k][j] += out[k][j]*conjf(out[k][j]);
	    }
	    if (crosspol & ((k/2)%2==1)) { // 2,3,7,8
	      l = k/4 + k%2;
	      for (j=0; j<npoint; j++) {
		cross[l][j] += out[k-2][j]*conjf(out[k][j]);
	      }
	    }
	  }
	  naver++;
	  
	  if (nint>0 && naver>=nint) {
	    char tmpstr[MAXSTR], *tmpchr;
	    
	    if (!first && pause) { /* Pause between plots */
	      printf("<Type Return to continue>\n");
	      tmpchr = fgets(tmpstr, MAXSTR, stdin);
	    }
	    doplot(nchan, chans, npoint, bchan, echan, naver, swapped, 
		   nx, domax,doavg, dolog, dooverplot, crosspol, ymax, ymin, 
		   xvals, spectrum, cross, plotspec, NULL, secperfft, mjd);
	    mjd += secperfft*naver/(60*60*24);
	    tB = tim();
	    printf("Integration look %.1f sec\n", tB-tA);
	    tA = tB;

	    naver = 0;
	    first = 0;
	    for (k=0; k<nchan; k++) {
	      if (!chans[k]) continue;
	      for (j=0; j<npoint; j++) {
		spectrum[k][j] = 0;
	      }
	    }
	  }
	}
	t1 = tim();
	tt += t1-t0;
      } // Loop over file
      close(file);

      if (online) break;
    } // Loop between files
    
    printf("Total computational time= %0.1f seconds for %d ffts\n", 
	   tt, nfft);
    tt = 0;
    nfft = 0;

    if (nint<0 || (float)naver/(float)nint>0.25) {

      if (dump) {
	// Remove leading path if present
	cptr = rindex(filename, '/');
	if (cptr==NULL) cptr = filename;
	strcpy(outfile, filename);

	// Remove trailing prefix, if present
	cptr = rindex(outfile, '.');
	if (cptr!=NULL) *cptr = 0;
	strcat(outfile, ".spec");

	printf("***** %s\n", outfile);
      } 

      doplot(nchan, chans, npoint, bchan, echan, naver, swapped, 
	     nx, domax, doavg, dolog, dooverplot, crosspol, ymax, ymin,
	     xvals, spectrum, cross, plotspec, NULL, secperfft, mjd);
      tB = tim();
      printf("DEBUG: Integration look %.1f sec\n", tB-tA);
      tA = tB;

      if (online) {
	naver = 0;
	first = 0;
	for (k=0; k<nchan; k++) {
	  if (!chans[k]) continue;
	  for (j=0; j<npoint; j++) {
	    spectrum[k][j] = 0;
	  }
	}
      }
      if (docommand) {
	int status;
	char cmdstr[MAXSTR+2];
	if (strlen(command)+strlen(outfile)>MAXSTR) {
	  fprintf(stderr, "Outfile and commnd strig too long\n");
	} else {
	  sprintf(cmdstr, command, outfile);
	  printf("Running: %s\n", cmdstr);
	  status = system(cmdstr);
	  if (status==-1) fprintf(stderr, "Could not execute command\n");
	}
      }
    }

    if (!online) break;

  } // Loop between files for online mode

  fftwf_free(mem);
  for (k=0; k<nchan; k++) {
    if (!chans[k]) continue;
    //fftwf_destroy_plan(plans[k]);
    //fftwf_free(in[k]); 
    //fftwf_free(out[k]);
  }
  fftwf_free(xvals);

  if (bits==2) {
    tcount = 0;
    for (i=0; i<4; i++) {
      tcount += lcount[i];
    }
    printf("Total counts %.1fe6\n", tcount/1e6);
    for (i=0; i<4; i++) {
      printf(" %5.1f", lcount[i]/(double)tcount*100);
    }
    printf("\n");
  }

  if (!domax && !doavg) cpgend();

  return(1);
}

int firstplot = 1;
void doplot(int nchan, int chans[], int npoint, int bchan, int echan, 
	    int naver, int swapped, int nx, int domax, int doavg, 
            int dolog, int dooverplot, int crosspol, float ymax,  float ymin,  
	    float xvals[], double *spectrum[MAXCHAN], fftwf_complex *cross[MAXCHAN/2],
	    float plotspec[], char outfile[], double secperfft, double mjd) {

  int j, k, m, q, hours, minutes, seconds;
  float max, min, delta;
  double sum;
  char msg[MAXSTR+1], str[MAXSTR];
  FILE *os = 0;

  if (outfile!=NULL) {
    os=fopen(outfile,"w");
    if (os==NULL) {
      fprintf(stderr, "Error opening output file %s\n", outfile);
      outfile = NULL;
    }
  }

  m = 0;
  for (k=0; k<nchan; k++) {
    if (!chans[k]) continue;
    
    if (dolog) 
      max = log10(spectrum[k][bchan] / naver/npoint/2/M_PI);
    else
      max = spectrum[k][bchan] / naver/npoint/2/M_PI;
    min = max;
    

    for (j=bchan; j<echan; j++) {
      plotspec[j] = spectrum[k][j] / naver/npoint/2/M_PI;
      if (dolog) plotspec[j] = log10(plotspec[j]);
      if (plotspec[j]>max) 
	max = plotspec[j];
      else if (plotspec[j]<min) 
	min = plotspec[j];
    }

    if (domax) {
      printf("%.1f", max);
    }

    if (doavg) {
      sum = 0;
      for (j=npoint*0.1;j<npoint*0.9;j++) {
	sum += spectrum[k][j] / naver/npoint/2/M_PI;
      }
      printf("%f ", sum/(npoint*0.8));
    }

    if (domax || doavg) continue;

    delta = (max-min)*0.05;
    min -= delta/2;
    max += delta;
    if (ymax != 0.0) max = ymax;
    if (ymin != 0.0) min = ymin;

    cpgsci(7);

    cpgbbuf();
    cpgpanl(m%nx+1,m/nx+1);
    if (firstplot) cpgeras();
    cpgvstd();

    if (firstplot) {
      if (swapped) 
	cpgswin(echan,bchan,min,max);
      else 
	cpgswin(bchan,echan,min,max);
      if (dolog)
	cpgbox("BCNST", 0.0, 0, "BCNSTL", 0.0, 0);
      else
	cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
    }

    sprintf(msg, "Channel %d", k+1);
    if (outfile!=NULL)
      fprintf(os,"%d  (number of channels for IF Channel %d)\n", 
	      echan-bchan,k+1);
    if (firstplot) {
      if (dolog)
	cpglab("Channel", "log\\d10\\u(Correlation)", msg);
      else
	cpglab("Channel", "Correlation", msg);
    }
    cpgsci(2);
    cpgline(echan-bchan, &xvals[bchan], &plotspec[bchan]);

    // Integration Time
    sprintf(str, "Tint = %.3f", naver*secperfft);
    cpgsci(7);
    cpgmtxt("T", 1.0, 1.0, 1.0, str);

    // Time
    seconds = floor(fmod(mjd, 1.0)*24*60*60+0.5);
    hours = seconds/3600;
    seconds -= hours*3600;
    minutes = seconds/60;
    seconds %= 60;
    sprintf(str, "%02d:%02d:%02d", hours, minutes, seconds);
    cpgmtxt("T", 1.0, 0.0, 0.0, str);

    if (outfile!=NULL)
      for (q=bchan;q<echan;++q)
	fprintf(os,"%5.0f    %f\n",xvals[q]-0.5,plotspec[q]);

    cpgebuf();

    m++;
  }

  if (outfile!=NULL) fclose(os);

  if (crosspol) {
    for (k=0; k<nchan/2; k++) {
      int l = k + (k/2)*2;
      if (!(chans[l]&&chans[l+2])) continue;

      // Amplitude
    
      max = abs(cross[k][bchan]) / naver/npoint/2/M_PI;
      min = max;
    
      for (j=bchan; j<echan; j++) {
	plotspec[j] = cross[k][j] / naver/npoint/2/M_PI;
	if (plotspec[j]>max) 
	  max = plotspec[j];
	else if (plotspec[j]<min) 
	  min = plotspec[j];
      }

      delta = (max-min)*0.05;
      min -= delta/2;
      max += delta;
      if (ymax != 0.0) max = ymax;
      cpgsci(7);

      cpgbbuf();
      cpgpanl(m%nx+1,m/nx+1);
      cpgeras();
      cpgvstd();

      cpgswin(bchan,echan,min,max);
  
      cpgbox("BCNST", 0.0, 0, "BNST", 0.0, 0);

      sprintf(msg, "Cross %d-%d", l, l+2);

      cpglab("Channel", "Correlation", msg);
      cpgsci(2);
      cpgline(echan-bchan, &xvals[bchan], &plotspec[bchan]);


      // Phase
    
      max = carg(cross[k][bchan]) /M_PI*180;
      min = max;
    
      for (j=bchan; j<echan; j++) {
	plotspec[j] = carg(cross[k][j]) / M_PI*180;
      }


      cpgswin(bchan,echan,-360,360);
  
      cpgsci(3);
      cpgline(echan-bchan, &xvals[bchan], &plotspec[bchan]);

      cpgebuf();

      m++;
    }
  }

  if (domax || doavg) printf("\n");

  if (dooverplot) firstplot = 0;

}

double cal2mjd(int day, int month, int year) {
  int m, y, c, x1, x2, x3;

  if (month <= 2) {
    m = month+9;
    y = year-1;
  } else {
    m = month-3;
    y = year;
  }

  c = y/100;
  y = y-c*100;

  x1 = 146097*c/4;
  x2 = 1461*y/4;
  x3 = (153*m+2)/5;

  return(x1+x2+x3+day-678882);
}

double tm2mjd(struct tm date) {
  int m, y, c, x1, x2, x3;
  double dayfrac;

  if (date.tm_mon < 2) {
    m = date.tm_mon+10;
    y = date.tm_mon+1900-1;
  } else {
    m = date.tm_mon-2;
    y = date.tm_year+1900;
  }

  c = y/100;
  y = y-c*100;

  x1 = 146097*c/4;
  x2 = 1461*y/4;
  x3 = (153*m+2)/5;

  dayfrac = ((date.tm_hour*60.0+date.tm_min)*60.0+date.tm_sec)
    /(60.0*60.0*24.0);

  return(cal2mjd(date.tm_mday, date.tm_mon+1, date.tm_year+1900)
	 +dayfrac);
}
