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
#include <signal.h>
#include <strings.h>
#include <math.h>
#include <arpa/inet.h>
#include <getopt.h>
#include <netdb.h>  
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include "../vsib/vheader.h"

#define FRAMESIZE       2500*4  // bytes
#define MK5BHEADSIZE  4         // 32bit words
#define MAXSTR        200 

#define DEBUG(x) 

double tm2mjd(struct tm);
void kill_signal (int);
unsigned int crcc (unsigned char *idata, int len, int mask, int cycl);
unsigned short reversebits16 (unsigned short s);
double cal2mjd(int day, int month, int year);
void init_bitreversal ();
int setup_net(int isserver, char *hostname, int port, int window_size, 
	      int *sock);
int netsend(int sock, char *buf, size_t len);
double tim(void);

void shuffle_init(float bandwidth, int bytespersample);
void atsignmag_init();
void correct_atsignmag(char *buf, size_t n);
void shuffle_data (char *p, size_t n, float bandwidth, int mode);
void make_littleendian(char *p, size_t n);


// Globals needed for signal handling
volatile int time_to_quit = 0;
volatile int sig_received = 0;

static unsigned char bytetable[256];
static unsigned short *shuffle_shorttable=NULL;
static unsigned char *shuffle_bytetable=NULL;
static uint8_t atsignmag_bytetable[256];

int main (int argc, char * const argv[]) {
  unsigned char j1, j2, j3, s1, s2, s3, s4, s5, crcdata[6];
  unsigned short *nframe, *crc, fnamesize;
  
  int framesize, nfile, infile, outfile, rate, framepersec, opt;
  int nread, status, nwrote, intmjd, secmjd, nsec, fracsec, tmp;
  int first, nupdate, sock;
  float ftmp;
  float bandwidth=-1;
  int bytespersample=0;
  double t0, t1;
  char outname[MAXSTR+1] = "";
  char outdir[MAXSTR+1] = "";
  char tmpstr[MAXSTR+1] = "";

  char *buf, *dotptr, msg[MAXSTR+50];
  u_int32_t mk5bheader[MK5BHEADSIZE];
  double startmjd, mjd;
  struct tm time;
  vhead *header;
  struct stat filestat;
  unsigned long long filesize, totalframe, totalsent;
  
  int offset = 0;
  int concat = 0;
  int net = 0;
  char postfix[MAXSTR+1] = ".m5b";
  int server = 0;
  float timeupdate = 50; /* Time every 50 MiB by default */
  int port = 52100;     /* TCP port to use */
  int window_size = -1;	
  char hostname[MAXSTR+1] = ""; /* Host name to send data to */

  struct option options[] = {
    {"postfix", 1, 0, 'P'},
    {"outdir", 1, 0, 'o'},
    {"offset", 1, 0, 'O'},
    {"dir", 1, 0, 'o'},
    {"outfile", 1, 0, 'f'},
    {"port", 1, 0, 'p'},
    {"host", 1, 0, 'H'},
    {"hostname", 1, 0, 'H'},
    {"time", 1, 0, 't'},
    {"window", 1, 0, 'w'},
    {"server", 0, 0, 's'},
    {"concat", 0, 0, 'c'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  init_bitreversal();
  atsignmag_init();

  first = 1;
  intmjd = 0;
  secmjd = 0;
  outfile = 0;

  /* Read command line options */
  while (1) {
    opt = getopt_long_only(argc, argv, "hp:o:f:H:", options, NULL);
    if (opt==EOF) break;

    switch (opt) {

    case 'O':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
	fprintf(stderr, "Bad offset value %s\n", optarg);
      else {
	offset = tmp;
      }
      break;
            
    case 'o':
      if (strlen(optarg)>MAXSTR) {
	fprintf(stderr, "Outdir too long\n");
	return(1);
      }
      strcpy(outdir, optarg);
      if (outdir[strlen(outdir)-1] == '/')  {// Remove trailing "/"
	outdir[strlen(outdir)-1] = 0;
      }
      break;

    case 'P':
      if (strlen(optarg)>MAXSTR) {
	fprintf(stderr, "Postfix too long\n");
	return(1);
      }
      strcpy(postfix, optarg);
      break;

    case 'f':
      if (strlen(optarg)>MAXSTR) {
	fprintf(stderr, "Outfile too long\n");
	return(1);
      }
      strcpy(outname, optarg);
      break;

    case 'w':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1)
	fprintf(stderr, "Bad window option %s\n", optarg);
      else 
	window_size = ftmp * 1024;
     break;
     
    case 'p':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
	fprintf(stderr, "Bad port option %s\n", optarg);
      else {
	port = tmp;
      }
      net = 1;
      break;
      
    case 't':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1 || ftmp <= 0.0)
	fprintf(stderr, "Bad time update option %s\n", optarg);
      else 
	timeupdate = ftmp;
      break;
      
    case 'H':
      if (strlen(optarg)>MAXSTR) {
	fprintf(stderr, "Hostname too long\n");
	return(1);
      }
      strcpy(hostname, optarg);
      net = 1;
      break;

    case 's':
      server = 1;
      net = 1;
      break;

    case 'c':
      concat = 1;
    break;

    case 'h':
      printf("Usage: lba2mk5b [options] <file> [<file> ...]\n");
      printf("  -o/outdir/dir <DIR>   Output directory for converted data\n");
      printf("  -c/concat             Concatenate into single output file\n");
      printf("  -O/offset <SEC>       Offset (in sec) to add to time\n");
      printf("  -f/outfile <NAME>     Outpit file name\n");
      printf("  -p/postfix <postfix>  Postfix for output files\n");
      printf("  -H/host <host>        Remote host to copy data\n");
      printf("  -p/port <port>        TCP port number\n");
      printf("  -w/window <window>    TCP window size (kb)\n");
      printf("  -s/server             Act as server rather than client\n");
      printf("  -t/time <time>        Interval between updates (MBytes)\n");
      printf("  -h/-help              This list\n");

      return(1);
    break;
    
    case '?':
    default:
      break;
    }
  }

  framesize = FRAMESIZE;
  buf = malloc(framesize);
  if (buf==NULL) {
    sprintf(msg, "Trying to allocate %d KB", framesize/1000);
    perror(msg);
    return(1);
  }

  if (strlen(hostname) == 0) {
    strcpy(hostname, "localhost");
  }
  nupdate = timeupdate/(framesize/1e6);

  if (strlen(outdir)>0) {
    printf("Writing data to %s\n", outdir);
  }

  header = newheader();

  // Assume data format is little endian
  
  // Mark5b Sync word is constant. Blank user bits

  mk5bheader[0] = 0xABADDEED;
  mk5bheader[1] = 0xF00F0000;
  mk5bheader[2] = 0x0;
  mk5bheader[3] = 0x0;
  // Use pointed into header for frame number and crc
  nframe = (unsigned short*)&mk5bheader[1];
  crc = (unsigned short*)&mk5bheader[3];

  if (net) {
    
    status = setup_net(server, hostname, port, window_size, &sock);
    if (status) exit(1);

    if (concat) { // Need to determine total size to transmit. We need to
                  // account for header, so must open each file
      unsigned long long totalfilesize = 0;

      for (nfile=optind; nfile<argc; nfile++) {
	infile = open(argv[nfile], OPENREADOPTIONS);
	if (infile==-1) {
	  sprintf(msg, "Failed to open input file (%s)", argv[nfile]);
	  perror(msg);
	  continue;
	}
	readheader(header, infile, NULL);

	status =  fstat(infile, &filestat);
	if (status != 0) {
	  sprintf(msg, "Trying to stat %s", argv[nfile]);
	  perror(msg);
	  return(1);
	}

	if (S_ISDIR(filestat.st_mode)) { // Skip directories
	  close(infile);
	  continue;
	}

	filesize = filestat.st_size - header->headersize;
	totalframe = (filesize+framesize-1)/framesize;
	totalfilesize += filesize+totalframe*MK5BHEADSIZE*4;

	close(infile);
      }
      filesize = totalfilesize;
    }
  }

  /* Install a ^C catcher */
  signal (SIGINT, kill_signal);

  t0 = tim(); /* So we can time average write per file */
  totalsent = 0;
  
  printf("Reading in chunks of %d kB\n", framesize/1000);
  for (nfile=optind; nfile<argc; nfile++) {

    infile = open(argv[nfile], OPENREADOPTIONS);
    if (infile==-1) {
      sprintf(msg, "Failed to open input file (%s)", argv[nfile]);
      perror(msg);
      continue;
    }
    printf("Reading %s\n", argv[nfile]);

    // Read lbadr header
    readheader(header, infile, NULL);

    /* Calculate the data rate, assuming Nyquist */
    rate = header->nchan*header->bandwidth*header->numbits*2;

    /* Mjd of start time */
    gettime(header, &time);
    startmjd = tm2mjd(time)+offset/(24.0*60.0*60.0);
    nsec = 0;

    if (header->bandwidth!=bandwidth) {
      bytespersample = header->nchan*header->numbits;
      bandwidth = header->bandwidth;
      if (bandwidth==64) bytespersample *=4;
      if (bandwidth==32) bytespersample *=2;
      bytespersample /= 8;
      if (bandwidth==64)
	shuffle_init(bandwidth, bytespersample);
    }

    *nframe = 0;
    framepersec = rate*1e6/8 / framesize;

    // Open output file

    // File name contained in buffer
    dotptr = rindex(argv[nfile], '.');

    if (dotptr!=NULL) {
      *dotptr = 0;
    }

    if (first && strlen(outname)>0) {
      if (strlen(outdir)>0) {
	if (strlen(outdir)+strlen(outname)+1 > MAXSTR) {
	  fprintf(stderr, "%s/%s too long. Increase \"MAXSTR(%d)\"\n", 
		  outdir, outname, MAXSTR);
	  return(1);
	}
	sprintf(tmpstr, "%s/%s", outdir, outname);
	strcpy(outname, tmpstr);
      }
    
    } else if (strlen(outdir)>0) {
      if (strlen(outdir)+strlen(argv[nfile])+1+strlen(postfix) > MAXSTR) {
      	fprintf(stderr, "%s/%s%s too long. Increase \"MAXSTR(%d)\"\n", 
      		outdir, argv[nfile], postfix, MAXSTR);
      	return(1);
      }

      sprintf(outname, "%s/%s%s", outdir, argv[nfile], postfix);

    } else {
      if (strlen(argv[nfile])+strlen(postfix) > MAXSTR) {
      	fprintf(stderr, "%s%s too long. Increase \"MAXSTR(%d)\"\n", 
      		argv[nfile], postfix, MAXSTR);
      	return(1);
       }

      strcpy(outname, argv[nfile]);
      strcat(outname, postfix);
    }

    if (net) {
      fnamesize = strlen(outname)+1;
      if (fnamesize==1) {
	printf("Skipping %s\n", argv[nfile]);
	close(infile);
	continue;
      }

      /* Fill first buffer with filesize and filename */
      /* 8 byte file size
	 2 byte filename size
	 n bytes filename 
      */

      if (!concat) { // Already calculated if concat
	status =  fstat(infile, &filestat);
	if (status != 0) {
	  sprintf(msg, "Trying to stat %s", argv[nfile]);
	  perror(msg);
	  return(1);
	}

	if (S_ISDIR(filestat.st_mode)) {
	  printf("Skipping %s (directory)\n", argv[nfile]);
	  close(infile);
	  continue;
	}
	
	filesize = filestat.st_size - header->headersize;
	totalframe = (filesize+framesize-1)/framesize;
	filesize += totalframe*MK5BHEADSIZE*4;
      }

      if (first || !concat) {
	status = netsend(sock, (char*)&filesize, sizeof(long long));
	if (status) {
	  printf("Netsend returned %d\n", status);
	  exit(1);
	}
	status = netsend(sock, (char*)&fnamesize, sizeof(short));
	if (status) {
	  printf("Netsend returned %d\n", status);
	  exit(1);
	}
	status = netsend(sock, outname, fnamesize);
	if (status) {
	  printf("Netsend returned %d\n", status);
	  exit(1);
	}
      }
    } else if (first || !concat) {
      outfile = open(outname, OPENWRITEOPTIONS, S_IRWXU|S_IRWXG|S_IRWXO); 
      if (outfile==-1) {
	sprintf(msg, "Failed to open output file (%s)", outname);
	perror(msg);
	continue;
      }
      printf("writing %s\n", outname);
    }
    
    first = 0;
    // Loop until EOF
    while (1) {

      if (time_to_quit) break;

      nread = read(infile, buf, framesize);

      if (nread==0) {  // EOF
	break;
      } else if (nread==-1) {
	perror("Error reading file");
	break;
      } else if (nread<framesize) {
	fprintf(stderr, "  Under-filled frame %d/%d\n", nread, framesize);
      }

      if (bandwidth==64) {
	shuffle_data(buf, nread, bandwidth, bytespersample);
      }

      //if (bytespersample==2) make_littleendian(buf, nread);

      // Create and write Mark5b header

      if (*nframe==0) { // Start of second, set first word of VLBA time code 
	if (nsec> 60*60*24) {
	  nsec = 0;
	  startmjd += 1.0;
	}
	mjd = startmjd + nsec*(1.0/(60*60*24));
	intmjd = (int)floor(mjd);
	secmjd = (int)floor((mjd-intmjd)*60*60*24+5e-7);  // Avoid round off errors

	// Word 2 VLBA BCD time code 'JJJSSSSS'. Only needs to be updated
	// once a second

	j1 = (intmjd%1000/100);
	j2 = ((intmjd%100)/10);
	j3 = (intmjd%10);
	
	s1 = (secmjd/10000)&0xF;
	s2 = ((secmjd%10000)/1000)&0xF;
	s3 = ((secmjd%1000)/100)&0xF;
	s4 = ((secmjd%100)/10)&0xF;
	s5 = (secmjd%10)&0xF;

	mk5bheader[2]  = j1<<28 | j2<<24 | j3<<20 | 
	  s1<<16 | s2<<12 | s3<<8 | s4<<4 | s5;

	crcdata[0] = j1<<4|j2;
	crcdata[1] = j3<<4|s1;
	crcdata[2] = s2<<4|s3;
	crcdata[3] = s4<<4|s5;
      }

      // MJD sec fraction
      fracsec = (*nframe/(double)framepersec)*10000;

      s1 = fracsec/1000;
      s2 = (fracsec%1000)/100;
      s3 = (fracsec%100)/10;
      s4 = fracsec%10;
      mk5bheader[3] = s1<<28 | s2<<24 | s3<<20 | s4<<16;

      // CRC
      crcdata[4] = s1<<4|s2;
      crcdata[5] = s3<<4|s4;
      *crc = reversebits16(crcc(crcdata, 48, 040003, 16));

      if (net) {
	status = netsend(sock, (char*)mk5bheader, MK5BHEADSIZE*4);
	if (status) {
	  perror("Netsend failed\n");
	  exit(1);
	}
	totalsent += MK5BHEADSIZE*4;
	status = netsend(sock, buf, nread);
	if (status) {
	  perror("Netsend failed\n");
	  exit(1);
	}
	totalsent += nread;
      } else {
	nwrote = write(outfile, mk5bheader, MK5BHEADSIZE*4);
	if (nwrote==-1) {
	  perror("Error writing outfile");
	  break;
	} else if (nwrote!=MK5BHEADSIZE*4) {
	  fprintf(stderr, "Warning: Did not write all bytes! (%d/%d)\n",
		  nwrote, MK5BHEADSIZE*4);
	  break;
	}

	// Write data
	nwrote = write(outfile, buf, nread);
	if (nwrote==-1) {
	  perror("Error writing outfile");
	  break;
	} else if (nwrote!=nread) {
	  fprintf(stderr, "Warning: Did not write all bytes! (%d/%d)\n",
		  nwrote, nread);
	  break;
	}
      }

      *nframe = (*nframe+1) % framepersec;
      if (*nframe==0) nsec++;
    } // Loop over input file

    /* Close the file */
    status = close(infile);
    if (status!=0) {
      perror("Error closing input file");
    }

    if (!concat && !net) {
      status = close(outfile);
      if (status!=0) {
	perror("Error closing output file");
      }
    }

    if (time_to_quit) break;
  }

  if (net) {
    float speed;

    status = close(sock);
    if (status!=0) {
      perror("Error closing socket");
    }

    t1 = tim(); /* So we can time average write per file */
    speed = totalsent/(t1-t0)*8/1e6;
    printf("   average send: %.1f Mbps\n", speed);

  } else if (concat) {
    status = close(outfile);
    if (status!=0) {
      perror("Error closing output file");
    }
  }

  /* A signal may have told us to quit. Raise this signal with the default
     handling */
    signal (sig_received, SIG_DFL);
  if (time_to_quit) raise (sig_received);
  
  return(0);
}

void kill_signal (int sig) {
  /* We may get called twice */
  if (time_to_quit) {
    fprintf(stderr, "kill_signal called second time\n");
    return;
  }
  time_to_quit = 1;
  sig_received = sig;

  signal (sig, kill_signal); /* Re-install ourselves to disable double signals */
}  


/* cal2mjd

 (double) mjd = cal2mjd(struct tm date);

 Converts a Unix tm struct (universal time) into a modified Julian day 
 number.
    date     UT time
    mjd     Modified Julian day (JD-2400000.5)

*/

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

// crc subroutine - converts input into a 16 bit CRC code
unsigned int crcc (unsigned char *idata, int len, int mask, int cycl) {
    
  unsigned int istate = 0;
  int idbit;
  int q, ich, icb;

  for (idbit = 1; idbit <= len; idbit++) {
    q = istate & 1;
    ich = (idbit - 1) / 8;
    icb = 7 - (idbit - 1) % 8;
    if ((((idata[ich] >> icb) & 1) ^ q)  ==  0)
      istate &= -2;
    else {
      istate ^=  mask;
      istate |= 1;
    }
    istate = (istate >> 1) | (istate & 1) << (cycl -1);
  } 
  return (istate);
}

unsigned short reversebits16 (unsigned short s) {
  unsigned char *c1, *c2;
  unsigned short ss;
  
  c1 = (unsigned char*)&s;
  c2 = (unsigned char*)&ss;
  
  // Flip bytes and bits
  c2[0] = bytetable[c1[1]]; 
  c2[1] = bytetable[c1[0]];
  return ss;
}

void init_bitreversal () {
  unsigned int i;
  for (i=0; i<256; i++) {
    bytetable[i] = ((i&0x80)>>7
                    |(i&0x40)>>5
                    |(i&0x20)>>3
                    |(i&0x10)>>1
                    |(i&0x08)<<1
                    |(i&0x04)<<3
                    |(i&0x02)<<5
                    |(i&0x01)<<7);
  }
}

int setup_net(int isserver, char *hostname, int port, int window_size, 
	      int *sock) {
  int ssock, status;
  unsigned long ip_addr;
  socklen_t client_len, winlen;
  struct hostent     *hostptr;
  struct sockaddr_in server, client;    /* Socket address */

  *sock = 0;

  memset((char *) &server, 0, sizeof(server));
  server.sin_family = AF_INET;
  server.sin_port = htons((unsigned short)port); /* Which port number to use */

  /* Create the initial socket */
  ssock = socket(AF_INET,SOCK_STREAM,0); 
  if (ssock==-1) {
    perror("Error creating socket");
    return(1);
  }

  if (window_size>0) {
    status = setsockopt(ssock, SOL_SOCKET, SO_SNDBUF,
			(char *) &window_size, sizeof(window_size));
    if (status!=0) {
      perror("Error setting socket send buffer");
      close(ssock);
      return(1);
    } 

    status = setsockopt(ssock, SOL_SOCKET, SO_RCVBUF,
			(char *) &window_size, sizeof(window_size));
    
    if (status!=0) {
      perror("Error setting socket receive buffer");
      close(ssock);
      return(1);
    }

    /* Check what the window size actually was set to */
    winlen = sizeof(window_size);
    status = getsockopt(ssock, SOL_SOCKET, SO_SNDBUF,
			(char *) &window_size, &winlen);
    if (status!=0) {
      close(ssock);
      perror("Getting socket options");
      return(1);
    }
    printf("Sending buffersize set to %d Kbytes\n", window_size/1024);
    
  }

  if (isserver) {

    /* Initialise server's address */
    server.sin_addr.s_addr = htonl(INADDR_ANY); /* Anyone can connect */
  
    status = bind(ssock, (struct sockaddr *)&server, sizeof(server));
    if (status!=0) {
      perror("Error binding socket");
      close(ssock);
      return(1);
    } 
  
    /* We are willing to receive conections, using the maximum
       back log of 1 */
    status = listen(ssock,1);
    if (status!=0) {
      perror("Error binding socket");
      close(ssock);
      return(1);
    }

    printf("Waiting for connection\n");

    /* Accept connection */
    client_len = sizeof(client);
    *sock = accept(ssock, (struct sockaddr *)&client, &client_len);
    if (*sock==-1) {
      perror("Error connecting to client");
      close(ssock);
      return(1);
    }
      
    printf("Got a connection from %s\n",inet_ntoa(client.sin_addr));

  } else {  // Acting as client

    hostptr = gethostbyname(hostname);
    if (hostptr==NULL) {
      fprintf(stderr,"Failed to look up hostname %s\n", hostname);
      close(ssock);
      return(1);
    }
  
    memcpy(&ip_addr, (char *)hostptr->h_addr, sizeof(ip_addr));
    server.sin_addr.s_addr = ip_addr;
  
    printf("Connecting to %s\n",inet_ntoa(server.sin_addr));

    status = connect(ssock, (struct sockaddr *) &server, sizeof(server));
    if (status!=0) {
      perror("Failed to connect to server");
      close(ssock);
      return(1);
    }
    *sock = ssock;
  }

  return(0);
}

int netsend(int sock, char *buf, size_t len) {
  char *ptr;
  int ntowrite, nwrote;
  
  ptr = buf;
  ntowrite = len;

  while (ntowrite>0) {
    nwrote = send(sock, ptr, ntowrite, 0);
    if (nwrote==-1) {
      if (errno == EINTR) continue;
      perror("Error writing to network");

      return(1);
    } else if (nwrote==0) {
      printf("Warning: Did not write any bytes!\n");
      return(1);
    } else {
      ntowrite -= nwrote;
      ptr += nwrote;
    }
  }
  return(0);
}

double tim(void) {
  struct timeval tv;
  double t;

  gettimeofday(&tv, NULL);
  t = (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;

  return t;
}

void shuffle_init(float bandwidth, int bytespersample) {
  int i, n;
  unsigned char *p;

  n = 0;

  if (bytespersample==2) {  // 16bit - 2 channel 64 MHz
    n = (1<<16)*sizeof(short);
    if (shuffle_shorttable!=NULL) free(shuffle_shorttable);
    shuffle_shorttable = malloc(n);
    if (shuffle_shorttable==NULL) {
      fprintf(stderr, "Failed to allocate memory for shuffle lookup table\n");
      exit(1);
    }
    p = (unsigned char*)shuffle_shorttable;

    for (i=0; i<(1<<16); i++) {
      shuffle_shorttable[i] = ((i>>6)&0x3) | (i&0x30) | ((i<<6)&0x300) | ((i<<12)&0x3000) |
	((i>>12)&0xC) | ((i>>6)&0xC0) | (i&0xC00) | ((i<<6)&0xC000);
    }

  } else if (bytespersample==1) { // 8 bit -1  channel 64 MHz
    n = 256;
    if (shuffle_bytetable!=NULL) free(shuffle_bytetable);
    shuffle_bytetable = malloc(n);
    if (shuffle_bytetable==NULL) {
      fprintf(stderr, "Failed to allocate memory for shuffle lookup table\n");
      exit(1);
    }
    p = (unsigned char*)shuffle_bytetable;

    for (i=0; i<256; i++) {
      shuffle_bytetable[i] = ((i>>6)&0x3) | ((i>>2)&0xC) | ((i<<2)&0x30) | ((i<<6)&0xC0);
    }
  } else {
    fprintf(stderr, "Error: Could not initialise shuffle table %d bytes per sample\n", bytespersample);
    return;
  }

  for (i=0; i<n; i++) {
    *p = atsignmag_bytetable[*p];
    p++;
  }
  return;
}

void shuffle_data(char *p, size_t n, float bandwidth, int bytespersample) {
  int i;
  if (bandwidth==64) {
    if (bytespersample==2) {
      unsigned short *s;
      if (n%2) {
	fprintf(stderr, "Can only shuffle even number of bytes (%d)\n", (int)n);
	exit(1);
      }

      s = (unsigned short*)p;
      for (i=0; i<n/2;i++) {
	*s = shuffle_shorttable[*s];
	s++;
      }
    } else if (bytespersample==1) {
      unsigned char *c;
      c = (unsigned char*)p;
      for (i=0; i<n;i++) {
	*c = shuffle_bytetable[*c];
	c++;
      }

    } else {
      fprintf(stderr, "Cannot shuffle 64 MHz bandwidth with %d bytes per sample\n", bytespersample);
      exit(1);
    }
  } else {
    fprintf(stderr, "Cannot shuffle bandwidth %.1f\n", bandwidth);
    exit(1);
  }

  return;
}

void make_littleendian(char *p, size_t n) {
  int i;
  uint16_t *ss;  

  if (n%2) {
    fprintf(stderr, "Must read 2 bytes words to fix endianess (read %d)\n", 
	    (int)n);
    exit(1);
  }


  ss = (uint16_t*)p;
  for (i=0; i< n/2; i++) {
    *ss=htons(*ss); // Big to little endian
    ss++;
  }
  return;
}

unsigned char sample_at2vlba(unsigned char c) {
  if (c==0) 
    return 2;
  else if (c==1)
    return 1;
  else if (c==2)
    return 0;
  else if (c==3)
    return 3;
  else {
    fprintf(stderr, "Error: at2vlba value too large %d\n", c);
    exit(1);
  }
  return(0);
}


void atsignmag_init() {
  int i, j;

  for (i=0; i<256; i++) {
    atsignmag_bytetable[i] = 0;

    for (j=0; j<4; j++) {
      atsignmag_bytetable[i] |= (sample_at2vlba((i>>(2*j))&0x3)<<(j*2));
    }
  }
  return;
}

