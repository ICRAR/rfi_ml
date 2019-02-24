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
//#include <values.h>
#include <time.h>
#include <sys/time.h>
#include <cpgplot.h>
#include <complex.h>
#include <fftw3.h>

#include "../vsib/vheader.h"

#define MALLOC fftwf_malloc
#define FREE fftwf_free

#define MIN(x,y)   ((x) < (y) ? (x) : (y))

#define MAX(x,y)   ((x) > (y) ? (x) : (y))

#define MAXCHAN 8

int doplot(int nchan, int chans[], int npoint, int bchan, int echan, 
	   int naver, int swapped, int inverse, int nx, float ymax, 
	   fftwf_complex *spectrum[MAXCHAN], char outfile[]);

int readbuf(int file, char *mem, int memsize, off_t *offset, int nskip, 
	    int bytesperint, char *filename);
void unpackdata (int i, char *mem, float *in[], int bits, int bandwidth, 
		 int nchan, int chans[], int npoint, int bytesperint,
		 float lookup[], long long lcount[]);

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
  int i, j, k, file1, file2, status, nread1, nread2, opt, memsize, nint;
  int naver, readshort, nfft, nx, ny, swapped, nplot, bytesperint, first, tmp;
  int icount;
  char msg[MAXSTR+1], *filename1, *filename2, *outfile, *cptr;
  char  *mem1, *mem2;
  off_t offset1, offset2;
  float *in1[MAXCHAN], *in2[MAXCHAN], lookup[4];
  float ftmp;
  fftwf_complex *out1[MAXCHAN], *out2[MAXCHAN], *spectrum[MAXCHAN];
  double t0, t1, tt;
  long long lcount1[4], lcount2[4], tcount;
  /* Do up to MAXCHAN subbands at the same time */ 
  fftwf_plan plan1[MAXCHAN], plan2[MAXCHAN]; 

  int verbose;       /* Prints whats happening */
  int inverse;       /* Plot delay spectrum */
  int dump;          /* Write output spectrum */
  int bits;          /* Number of bits/sample */
  int nchan;         /* Total number of channels */
  int bandwidth;     /* Observing bandwidth (different byte format for 32 and 
			64 MHz data*/
  int nskip;         /* Number of bytes to skip at start of file */
  int npoint;        /* Number of spectral channels */
  float integration; /* Integration time, in seconds */
  int chans[MAXCHAN];/* Channels to correlate */
  int bchan;         /* First spectral point to plot */
  int echan;         /* Last spectral point to plot */
  float ymax;        /* Maximum yvalue of plot */
  encodingtype encoding; /* VLBA or AT bit encoding (offset binary or sign/mag) */
  char pgdev[MAXSTR+1] = "/xs"; /* Pgplot device to use */

  int chan1, chan2;
  vhead *header1, *header2;

  struct option options[] = {
    {"vlba", 0, 0, 'v'},
    {"npoint", 1, 0, 'n'},
    {"chan", 1, 0, 'C'},
    {"integration", 1, 0, 'i'},
    {"skip", 1, 0, 's'},
    {"if1", 1, 0, '1'},
    {"if2", 1, 0, '2'},
    {"sp1", 1, 0, 'q'},
    {"sp2", 1, 0, 'r'},
    {"ymax", 1, 0, 'y'},
    {"dump", 0, 0, 'D'},
    {"max", 0, 0, 'm'},
    {"device", 1, 0, 'd'},
    {"pgdev", 1, 0, 'd'},
    {"inverse", 0, 0, 'I'},
    {"verbose", 0, 0, 'V'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  verbose = 0;
  inverse = 0;
  dump = 0;
  nskip = 0;
  npoint = 128;
  encoding = AT;
  chan1 = -1;
  chan2 = -1;
  integration = -1.0;
  nint = -1;
  for (k=0;k<MAXCHAN;k++) chans[k]=0;
  bchan = -1;
  echan = -1;
  ymax = 0.0;
  filename1 = NULL;
  filename2 = NULL;
  icount = 0;

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

    case 'y':
      printf("Got ymax = %s\n", optarg);
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1)
 	fprintf(stderr, "Bad -ymax option %s\n", optarg);
      else 
	ymax = ftmp;
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

    case 'i':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1)
 	fprintf(stderr, "Bad -integration option %s\n", optarg);
      else 
	integration = ftmp;
      break;

    case 'd':
      if (strlen(optarg)>MAXSTR) {
	fprintf(stderr, "PGDEV option %s too long\n", optarg);
	return 0;
      }
      strcpy(pgdev, optarg);
      break;

    case 'V':
      verbose = 1;
      break;

    case 'D':
      dump = 1;
      break;

    case 'I':
      inverse = 1;
      break;

    case 'v':
      encoding = VLBA;
      break;

    case 'h':
      printf("Usage: vsib_checker [options]\n");
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

  if (argc-optind!=2) {
    fprintf(stderr, "Usage: fcross [options] file1 file2\n");
    exit(1);
  }


  /* Setup the lookup table */
  if (encoding==VLBA) {
    lookup[0] = +3.0;
    lookup[1] = +1.0;
    lookup[2] = -1.0;
    lookup[3] = -3.0;
  } else { /* AT */
    lookup[0] = +1.0;
    lookup[1] = -1.0;
    lookup[2] = +3.0;
    lookup[3] = -3.0;
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

  header1 = newheader();
  header2 = newheader();

  /* Open input files */
  filename1 = argv[argc-2];
  filename2 = argv[argc-1];
  file1 = open(filename1, OPENOPTIONS);
  if (file1==-1) {
    sprintf(msg, "Failed to open input file (%s)", filename1);
    perror(msg);
    //fftwf_free(mem);
    return(1);
  }
  file2 = open(filename2, OPENOPTIONS);
  if (file2==-1) {
    sprintf(msg, "Failed to open input file (%s)", filename2);
    perror(msg);
    //fftwf_free(mem);
    return(1);
  }

  printf("Processing %s & %s\n", filename1, filename2);

  /* Read the headers and sanity check*/
  readheader(header1, file1, NULL);
  readheader(header2, file2, NULL);

  //lseek(file1, 0, SEEK_CUR);

  if (header1->nchan!=header2->nchan 
      || header1->numbits!=header2->numbits 
      || header1->bandwidth!=header2->bandwidth) {
    fprintf(stderr, "Input files not consistent\n");
    exit(1);
  }

  nchan = header1->nchan;
  bits = header1->numbits;
  bandwidth = header1->bandwidth;

  if (integration>0) { // Calculate number of ffts per integration
    nint = integration/(npoint*2)*bandwidth*2*1e6;
    printf("Integrating over %d ffts\n", nint);
  }
	
  /* Channel selection sanity */
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

  /* Spectral point selection */
  if (bchan<0) bchan = 0;
  if (echan<0) echan = npoint-1;
  if (bchan > npoint-1) bchan = npoint-1;
  if (echan > npoint-1) echan = npoint-1;
  if (bchan==echan) {
    fprintf(stderr, "Warning: Odd channel range. Resetting\n");
    bchan = 0;
    echan = npoint-1;
  }
  
  swapped = 0;
  if (bchan>echan) {
    swapped = bchan;
    bchan = echan;
    echan = swapped;
    swapped = 1;
    fprintf(stderr, "Swapping frequency axis\n");
  } 
  echan++;
	
  /* Bytes per integration */
  if (bits==10) {
    bytesperint = npoint*2*16*nchan/8;
  } else {
    bytesperint = npoint*2*bits*nchan/8;
  }

  printf("Bytes per integration = %d\n", bytesperint);


  /* Memory buffers for file read */
  memsize = floor(BLOCKSIZE/bytesperint)*bytesperint+0.1;
  mem1 = MALLOC(memsize);
  mem2 = MALLOC(memsize);
  if (mem1==NULL || mem2==NULL) {
    sprintf(msg, "Trying to allocate %d byes", memsize);
    perror(msg);
    return(1);
  }

  /* Initiate fftw */
	
  for (k=0; k<nchan; k++) {
    if (!chans[k]) continue;

    in1[k] = MALLOC(sizeof(float) * npoint*2);
    in2[k] = MALLOC(sizeof(float) * npoint*2);
    if (in1[k]==NULL || in2[k]==NULL) {
      perror("Allocating memory");
      exit(1);
    }

    out1[k] = MALLOC(sizeof(fftwf_complex)*npoint+1);
    out2[k] = MALLOC(sizeof(fftwf_complex)*npoint+1);
    if (out1[k]==NULL || out2[k]==NULL) {
      perror("Allocating memory");
      exit(1);
    }

    spectrum[k] = MALLOC(sizeof(fftwf_complex)*npoint+1);
    if (spectrum[k]==NULL) {
      perror("Allocating memory");
      exit(1);
    }
    for (j=0; j<npoint; j++) {
      spectrum[k][j] = 0.0;
    }

    plan1[k] = fftwf_plan_dft_r2c_1d(npoint*2, in1[k], out1[k], 0);
    plan2[k] = fftwf_plan_dft_r2c_1d(npoint*2, in2[k], out2[k], 0);
  }

  /* How many plots to do */;
  ny = ceil(sqrt(nplot));
  nx = (nplot+ny-1)/ny;
	
  if (cpgbeg(0,pgdev,nx,ny)!=1) {
    fprintf(stderr, "Error calling PGBEGIN");
    return(1);
  }
  cpgask(0);
    
  for (i=0; i<4; i++) {
    lcount1[i] = 0;
    lcount2[i] = 0;
  }

  offset1 = 0;
  offset2 = 0;
  readshort = 0;
  
  if (nskip>0) {
    offset1 = lseek(file1, nskip, SEEK_CUR);
    if (offset1==-1) {
      perror(msg);
      //fftwf_free(mem);
      return(1);
    }
    offset2 = lseek(file2, nskip, SEEK_CUR);
    if (offset2==-1) {
      perror(msg);
      //fftwf_free(mem);
      return(1);
    }
  }

  /* Loop over file reading mulitple fft worth of data at a time */
  while (1) {
    nread1 = readbuf(file1, mem1, memsize, &offset1, nskip, bytesperint, filename1);
    if (nread1<0) {
      exit(1);
    } 
    nread2 = readbuf(file2, mem2, memsize, &offset2, nskip, bytesperint, filename1);
    if (nread2<0) {
      exit(1);
    } 

    if (nread1!=nread2) {
      fprintf(stderr, "Read inconsistent amounts of data from files\n");
      break;
    }

    if (nread1==0 || nread2==0)
      break;

    t0 = tim();
        
    /* The following loops decode the data into a set of float arrays */

    /* Copy data into "in" array, fft then accumulate */
    for (i=0; i<nread1/bytesperint; i++) {

      unpackdata(i, mem1, in1, bits, bandwidth, nchan, chans, npoint, 
		 bytesperint, lookup, lcount1);
      unpackdata(i, mem2, in2, bits, bandwidth, nchan, chans, npoint, 
		 bytesperint, lookup, lcount2);

      for (k=0; k<nchan; k++) {
	if (!chans[k]) continue;
	
	fftwf_execute(plan1[k]);
	fftwf_execute(plan2[k]);
	nfft++;
	for (j=0; j<npoint; j++) {
	  spectrum[k][j] += out1[k][j]*conjf(out2[k][j]);
	}
      }
      naver++;
      
      if (nint>0 && naver>=nint) {
	
	if (dump) {
	  // Remove leading path if present
	  cptr = rindex(filename1, '/');
	  if (cptr==NULL) cptr = filename1;
	  strcpy(outfile, filename1);
      
	  // Remove trailing prefix, if present
	  cptr = rindex(outfile, '.');
	  if (cptr!=NULL) *cptr = 0;
	  strcat(outfile, ".spec");

	  // Add a integration number
	  icount++;
	  sprintf(outfile, "%s.%d", outfile, icount);
	} 

	status = doplot(nchan, chans, npoint, bchan, echan, naver, swapped, 
			inverse, nx, ymax, spectrum, outfile);
	if (status) exit(1);
	
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

  printf("Total computational time= %0.1f seconds for %d ffts\n", 
	 tt, nfft);
  
  if (nint<0 || (float)naver/(float)nint>0.25) {
    
    if (dump) {
      // Remove leading path if present
      cptr = rindex(filename1, '/');
      if (cptr==NULL) cptr = filename1;
      strcpy(outfile, filename1);
      
      // Remove trailing prefix, if present
      cptr = rindex(outfile, '.');
      if (cptr!=NULL) *cptr = 0;
      strcat(outfile, ".spec");
    } 
    
    status = doplot(nchan, chans, npoint, bchan, echan, naver, swapped, 
		    inverse, nx, ymax, spectrum, outfile);
    if (status) exit(1);
  }

  FREE(mem1);
  FREE(mem2);
  for (k=0; k<nchan; k++) {
    if (!chans[k]) continue;
    //fftwf_destroy_plan(plans[k]);
    //fftwf_free(in[k]); 
    //fftwf_free(out[k]);
  }

  if (bits==2) {
    tcount = 0;
    for (i=0; i<4; i++) {
      tcount += lcount1[i];
    }
    printf("Total counts %.1fe6\n", tcount/1e6);
    for (i=0; i<4; i++) {
      printf(" %5.1f", lcount1[i]/(double)tcount*100);
    }
    printf("\n");

    tcount = 0;
    for (i=0; i<4; i++) {
      tcount += lcount2[i];
    }
    for (i=0; i<4; i++) {
      printf(" %5.1f", lcount2[i]/(double)tcount*100);
    }
    printf("\n");

  }

  cpgend();

  return(1);
}

int doplot(int nchan, int chans[], int npoint, int bchan, int echan, 
	   int naver, int swapped, int inverse, int nx,  float ymax,
	   fftwf_complex *spectrum[MAXCHAN], char outfile[]) {
  int j, k, m;
  float xmin, xmax, max, min, delta, *plotspec1, *plotspec2, *xvals;
  char msg[MAXSTR+1];
  fftwf_complex *out = NULL;
  fftwf_plan plan; 
  FILE *os = NULL;

  /* Initiate plotting arrays */

  xvals = MALLOC(sizeof(float)*npoint);
  plotspec1 = MALLOC(sizeof(float)*npoint);
  plotspec2 = MALLOC(sizeof(float)*npoint);
  if (xvals==NULL || plotspec1==NULL || plotspec2==NULL) {
    sprintf(msg, "Trying to allocate %d bytes", (int)sizeof(float)*npoint);
    perror(msg);
    return(1);
  }

  if (inverse) {
    out = MALLOC(sizeof(fftwf_complex)*npoint+1);

    if (out==NULL) {
      sprintf(msg, "Trying to allocate %d bytes", 
	      (int)sizeof(fftwf_complex)*npoint+1);
      perror(msg);
      return(1);
    }

    for (j=0; j<npoint; j++) {
      xvals[j] = j-npoint/2;
    }
  } else {
    for (j=0; j<npoint; j++) {
      xvals[j] = j+0.5;
    }
  }

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

    for (j=0; j<npoint; j++) spectrum[k][j] /= naver*npoint;

    if (inverse) {
      plan = fftwf_plan_dft_1d(npoint, spectrum[k], out, FFTW_BACKWARD, 
				   FFTW_ESTIMATE);
      fftwf_execute(plan);
      
      for (j=bchan; j<echan; j++) {
	int i = (j+npoint/2)%npoint;

	//plotspec1[i] = crealf(out[j]);
	//plotspec2[i] = cimagf(out[j]);

	plotspec1[i] = cabsf(out[j]);
	plotspec2[i] = cargf(out[j])/2/M_PI*360.0;
	

      }

      fftwf_destroy_plan(plan);

    } else {
      for (j=bchan; j<echan; j++) {
	//plotspec1[j] = crealf(spectrum[k][j]);
	//plotspec2[j] = cimagf(spectrum[k][j]);

	plotspec1[j] = cabsf(spectrum[k][j])*10;
	plotspec2[j] = cargf(spectrum[k][j])/2/M_PI*360.0;
      }
    }

    max = plotspec1[bchan];
    min = max;

    for (j=bchan; j<echan; j++) {
      if (plotspec1[j]>max) 
	max = plotspec1[j];
      else if (plotspec1[j]<min) 
	min = plotspec1[j];
      if (plotspec2[j]>max) 
	max = plotspec2[j];
      else if (plotspec2[j]<min) 
	min = plotspec2[j];
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

    if (inverse) {
      xmin = -npoint/2;
      xmax = npoint/2;
    } else {
      xmin = bchan;
      xmax = echan;
    }

    if (swapped) 
      cpgswin(xmin,xmax,min,max);
    else 
      cpgswin(xmax,xmin,min,max);
  
    cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);

    sprintf(msg, "Channel %d", k+1);
    if (outfile!=NULL)
      fprintf(os,"%d  (number of channels for IF Channel %d)\n", 
	      echan-bchan,k+1);
    cpglab("Channel", "Correlation", msg);
    cpgsci(2);
    cpgline(echan-bchan, &xvals[bchan], &plotspec1[bchan]);
    cpgsci(3);
    //cpgline(echan-bchan, &xvals[bchan], &plotspec2[bchan]);
    cpgpt(echan-bchan, &xvals[bchan], &plotspec2[bchan], 17);

    if (outfile!=NULL)
      for (j=bchan;j<echan;j++)
	fprintf(os,"%5.0f  %f %f\n",xvals[j]-0.5,plotspec1[j],plotspec2[j]);

    cpgebuf();

    m++;
  }
  if (outfile!=NULL)
    fclose(os);

  FREE(plotspec1);
  FREE(plotspec2);
  FREE(xvals);
  if (inverse) FREE(out);

  return(0);
}

int readbuf(int file, char *mem, int memsize, off_t *offset, int nskip, 
	    int bytesperint, char *filename) {
  int nread;
  off_t off;

  nread = read(file, mem, memsize);
  if (nread==0) {
    if (*offset<=nskip) {
      fprintf(stderr, "No data read from %s\n", filename);
      close(file);
      return(-1);
    }
    close(file);
    return(0);
  } else if (nread==-1) {
    perror("Error reading file");
    close(file);
    return(-1);
  } else if (nread%bytesperint != 0) { /* Need to read multiple of lags */
    /* Data may be pack with multiple samples/byte or mutiple 
       bytes/sample */
    int shift;
    shift = nread % bytesperint;
      
    off = lseek(file, -shift, SEEK_CUR); 
    if (off==-1) {
      perror("");
      return(-1);
    }
    nread -= shift;
  }
  
  if (nread<bytesperint) return(0);

  *offset += nread;
  return(nread);
}

void unpackdata (int i, char *mem, float *in[], int bits, int bandwidth, 
		int nchan, int chans[], int npoint, int bytesperint,
		float lookup[], long long lcount[]) {
  int j, k, l;
  register int samp;
  unsigned short *mem16;
  unsigned short *sample16;
  char *sample;

  mem16 = (unsigned short*)mem;

  if (bits==2) {
    if (bandwidth==64) { /* 64 MHz needs to be handle separately */
      if (nchan==2) {
	for (j=0; j<npoint/2; j++) {
	  for (k=0; k<nchan; k++) {
	    if (!chans[k]) continue;
	    samp = mem[i*bytesperint+j*2+k];
	    for (l=0; l<4; l++) {
	      in[k][j*4+l] = lookup[(samp>>(l*2))&0x3];
	      lcount[(samp>>(l*2))&0x3]++;
	    }
	  }
	}
      } else if (nchan==1) {
	for (j=0; j<npoint/2; j++) {
	  samp = mem[i*bytesperint+j];
	  for (l=0; l<4; l++) {
	    in[0][j*4+l] = lookup[(samp>>(l*2))&0x3];
	    lcount[(samp>>(l*2))&0x3]++;
	  }
	}
      } else {
	fprintf(stderr, "64 MHz %d channels not supported\n", nchan);
	exit(1);
      }
      
    } else if (bandwidth==32) {
      if (nchan==2) {
	for (j=0; j<npoint; j++) {
	  samp = mem[i*bytesperint+j];
	  for (k=0; k<nchan; k++) {
	    if (!chans[k]) continue;
	    in[k][j*2] = lookup[(samp>>(k*2))&0x3];
	    in[k][j*2+1] = lookup[(samp>>(k*2+2))&0x3];
	    lcount[(samp>>(k*2))&0x3]++;
	    lcount[(samp>>(k*2+2))&0x3]++;
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
	  samp = mem[i*bytesperint+j];
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
	  samp = mem[i*bytesperint+j];
	  
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
    sample =  &mem[i*bytesperint];
    sample--;
    for (j=0; j<npoint*2; j++) {
      for (k=0; k<nchan; k++) {
	sample++;
	if (!chans[k]) continue;
	in[k][j] = *sample;
      }
    }
  } else if (bits==10) {
    sample16 =  &mem16[i*bytesperint/2];
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
  } else {
    fprintf(stderr, "Unsupported number of bits %d\n", bits);
    exit(1);
  }
}    
