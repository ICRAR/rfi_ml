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
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <getopt.h>
#include <netdb.h>  
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <errno.h>

#include "vdif.h"

#define DEFAULTFRAMESIZE 10000
#define READBUFSIZE  2100000  // ~2MB
#define MAXSTR       255 

#define DEBUG(x) 

//double tm2mjd(struct tm);
void kill_signal (int);
int cal2mjd(int day, int month, int year);
int setup_net(int isserver, char *hostname, int port, int window_size, 
	      int *sock);
int netsend(int sock, char *buf, size_t len);
double tim(void);
void convertsamples(char *dataarray, char *orig, int framesize, int packetsize, 
		    int invert, int *ndone, int *nused);

// Globals needed for signal handling
volatile int time_to_quit = 0;
volatile int sig_received = 0;

int main (int argc, char * const argv[]) {
  int origframesize, nfile, infile, outfile, datarate, opt;
  int nread, status, nwrote, intmjd, tmp;
  int first, nupdate, sock, bufsize, startpacket;
  float ftmp;
  float bandwidth=-1;
  double t0, t1;
  vdif_header vheader;
  char outname[MAXSTR+1] = "";
  char outdir[MAXSTR+1] = "";
  char tmpstr[MAXSTR+1] = "";

  char *buf, *p, *dotptr, msg[MAXSTR+50];
  char *dataarray;
  int64_t mjdsec;
  //struct tm time;
  //vhead *header;
  //struct stat filestat;
  unsigned long long totalsent;

  int nyquist, bits, nif, npol, samplesize, packetsize, packetperread;
  int year, month, day, hour, minute, second, skip;
  int ndone, nused;
  long offset = 0;
  long offsetframes = 0;
  int concat = 0;
  int invert = 0;
  int net = 0;
  int framesize = DEFAULTFRAMESIZE;
  char postfix[MAXSTR+1] = ".vdif";
  int server = 0;
  float timeupdate = 50; /* Time every 50 MiB by default */
  int port = 52100;     /* TCP port to use */
  int window_size = -1;	
  char hostname[MAXSTR+1] = ""; /* Host name to send data to */

  struct option options[] = {
    {"postfix", 1, 0, 'P'},
    {"outdir", 1, 0, 'o'},
    //{"offset", 1, 0, 'O'},
    {"framesize", 1, 0, 'F'},
    {"dir", 1, 0, 'o'},
    {"outfile", 1, 0, 'f'},
    {"port", 1, 0, 'p'},
    {"host", 1, 0, 'H'},
    {"hostname", 1, 0, 'H'},
    {"time", 1, 0, 't'},
    {"window", 1, 0, 'w'},
    {"server", 0, 0, 's'},
    {"concat", 0, 0, 'c'},
    {"invert", 0, 0, 'i'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  first = 1;
  intmjd = 0;
  outfile = 0;

  /* Read command line options */
  while (1) {
    opt = getopt_long_only(argc, argv, "hp:o:f:H:F:", options, NULL);
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
            
    case 'F':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1)
	fprintf(stderr, "Bad framesize value %s\n", optarg);
      else {
	framesize = tmp;
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

    case 'i':
      invert = 1;
    break;

    case 'h':
      printf("Usage: dfb2vdif [options] <file> [<file> ...]\n");
      printf("  -o/outdir/dir <DIR>   Output directory for converted data\n");
      printf("  -c/concat             Concatenate into single output file\n");
      printf("  -c/invert             Invert the spectrum, ie LSB->USB\n");
      //printf("  -O/offset <SEC>       Offset (in sec) to add to time\n");
      printf("  -F/framesize <SIZE>   VDIF frame size\n");
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

  origframesize = framesize;
  framesize -= VDIFHEADERSIZE; // Remove size of header  if (framesize%8) framesize -= framesize%8;


  // Should read these from file
  bandwidth = 16;
  packetsize = 4160;
  nyquist = 2;
  bits = 2;
  nif = 1;
  npol = 2;


  // Make sure frame size is sensible
  samplesize = npol*nif*bits*2/8; // Complex data
  if (samplesize>1) {
    if (samplesize>8 || 8%samplesize) {
      fprintf(stderr, "Error: Cannot handle combination of ifs/bits/pols\n");
      exit(1);
    }

  } else {
    samplesize = 1;  // Catch multiple samples/byte
  }

  // Must fit integral number of samples

  datarate = bandwidth*1e6*nyquist*bits*nif*npol/8;

  while (framesize>0 && datarate%framesize) framesize -= 8;

  if (framesize <=0) {
    fprintf(stderr, "Error: Cannot fit frames into on second\n");
    exit(1);
  }

  printf("Using framesize %d (%d datasize)\n", framesize+VDIFHEADERSIZE, framesize);

  /* Get integral number of DFB packets per read */
  packetperread = READBUFSIZE/packetsize;
  bufsize = packetperread*packetsize;

  buf = malloc(bufsize);
  if (buf==NULL) {
    sprintf(msg, "Trying to allocate %d KB", bufsize/1024);
    perror(msg);
    return(1);
  }

  dataarray = malloc(framesize);
  if (dataarray==NULL) {
    sprintf(msg, "Trying to allocate %d bytes", framesize);
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

  if (net) {
    
    status = setup_net(server, hostname, port, window_size, &sock);
    if (status) exit(1);

  }

  /* Install a ^C catcher */
  signal (SIGINT, kill_signal);

  t0 = tim(); /* So we can time average write per file */
  totalsent = 0;


  vdif_createheader(&vheader, framesize+VDIFHEADERSIZE, datarate/framesize, 
		    0, bits, nif*npol, 1, "Pa");

  printf("Reading in chunks of %d kB\n", bufsize/1024);
  for (nfile=optind; nfile<argc; nfile++) {

    infile = open(argv[nfile], OPENREADOPTIONS);
    if (infile==-1) {
      sprintf(msg, "Failed to open input file (%s)", argv[nfile]);
      perror(msg);
      return(1);
    }
    printf("Reading %s\n", argv[nfile]);

    // Skip header
    status = lseek(infile, 4096, SEEK_SET);
    if (status<0) {
      perror("Skipping over header");
      exit(1);
    }

    // Extract time
    status = sscanf(argv[nfile], "%4d-%2d-%2d-%2d:%2d:%2d_%ld.000000.dada",
		    &year, &month, &day, &hour, &minute, &second, &offset);

    if (status<=0) {
      fprintf(stderr, "Error extracting time from %s\n", argv[nfile]);
      exit(1);
    }

    
    if (offset%framesize>0) {
      skip = framesize-(offset%framesize);
      offset += skip;
      startpacket = skip/packetsize;
      nused = (skip%packetsize)/2;
    } else {
      startpacket = 0;
      nused = 0;
    }

    mjdsec = ((cal2mjd(day, month, year)*24LL+hour)*60LL+minute)*60LL+second;

    // offset is guarenteed to be an integral number of VDIF frames at this stage
    offsetframes = offset/framesize;
    mjdsec += offsetframes/vheader.framepersec;

    status = vdif_setmjdsec(&vheader, mjdsec);
    if (status!=VDIF_NOERROR) {
      fprintf(stderr, "Setting VDIF time failed (%d)\n", status);
      return(status);
    }

    vheader.frame = offsetframes % vheader.framepersec;

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

    if (!net && (first || !concat)) {
      outfile = open(outname, OPENWRITEOPTIONS, S_IRWXU|S_IRWXG|S_IRWXO); 
      if (outfile==-1) {
	sprintf(msg, "Failed to open output file (%s)", outname);
	perror(msg);
	return(1);
      }
      printf("writing %s\n", outname);
    }

    
    first = 0;
    ndone = 0;
    // Loop until EOF
    while (1) {

      if (time_to_quit) break;

      nread = read(infile, buf, bufsize);

      if (nread==0) {  // EOF
	break;
      } else if (nread==-1) {
	perror("Error reading file");
	break;
      } else if (nread%packetsize) {
	fprintf(stderr, "Error: Did not read full packet\n");
      }

      int i;
      for (i=startpacket; i<packetperread; i++) {
	p = &buf[i*packetsize];

	while (1) {
	  convertsamples(dataarray, p, framesize, packetsize, 
			 invert, &ndone, &nused);

	  if (ndone>=framesize) { // Filled a frame

	    if (net) {
	      status = netsend(sock, (char*)&vheader, VDIFHEADERSIZE);
	      if (status) {
		perror("Netsend failed\n");
		exit(1);
	      }
	      totalsent += VDIFHEADERSIZE;

	      status = netsend(sock, dataarray, framesize);
	      if (status) {
		perror("Netsend failed\n");
		exit(1);
	      }
	      totalsent += nread;

	    } else { // disk
	      nwrote = write(outfile, &vheader, VDIFHEADERSIZE);
	      if (nwrote==-1) {
		perror("Error writing outfile");
		break; // BUG NEEDS TO BREAK OUT OF NEXT LOOP ALSO
	      } else if (nwrote!=VDIFHEADERSIZE) {
		fprintf(stderr, "Warning: Did not write all bytes! (%d/%d)\n",
			nwrote, VDIFHEADERSIZE);
		break;  // BUG NEEDS TO BREAK OUT OF NEXT LOOP ALSO
	      }

	      // Write data
	      nwrote = write(outfile, dataarray, framesize);
	      if (nwrote==-1) {
		perror("Error writing outfile");
		break; // BUG NEEDS TO BREAK OUT OF NEXT LOOP ALSO
	      } else if (nwrote!=framesize) {
		fprintf(stderr, "Warning: Did not write all bytes! (%d/%d)\n",
			nwrote, framesize);
		break;  // BUG NEEDS TO BREAK OUT OF NEXT LOOP ALSO
	      }
	    }

	    vdif_nextheader(&vheader);
	    ndone = 0;

	  }

	  if (nused>=packetsize/2) break;
	} // Loop over read packet
	nused= 0;
      } // Loop over read
      startpacket = 0;
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
  } // Loop over files

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

int cal2mjd(int day, int month, int year) {
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
    printf("Sending buffersize set to %d Kbytes\n", winlen/1024);
    
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

void convertsamples(char *dataarray, char *orig, int framesize, int packetsize, 
		    int invert, int *ndone, int *nused) {
  
  char *a,*b,*c, mask;

  // Assume single channel dual pol for now

  a = &orig[*nused];
  b = &orig[*nused]+packetsize/2;
  c = &dataarray[*ndone];

  if (invert)
    mask = 0x99;  // 2's complement to offset binary and invert imag sign
  else
    mask = 0xAA; // 2's complement to offset binary

  while (*ndone<framesize && *nused<packetsize/2) {

    *c = (((*a>>4)&0xF) | (*b&0xF0))^mask;
    c++;
    *c = ((*a&0xF) | ((*b<<4)&0xF0))^mask;
    a++; b++; c++;
    *ndone +=2;
    (*nused)++;
  }
  return;
}
