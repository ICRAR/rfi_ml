#ifdef __APPLE__

#define OSX

#define OPENOPTIONS O_WRONLY|O_CREAT|O_TRUNC

#else

#define LINUX
#define _LARGEFILE_SOURCE 
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64

#define OPENOPTIONS O_WRONLY|O_CREAT|O_TRUNC|O_LARGEFILE

#endif

#include <sys/types.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>     
#include <sys/timeb.h>
#include <sys/time.h>       
#include <sys/socket.h>  
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <string.h>
#include <signal.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <fcntl.h>
#include <getopt.h>

typedef struct disktime {
  double         time;
  unsigned long  size;
} disktime;

#define NBUF       3  // Must be at least 3
#define DEFAULT_BUFSIZE       2
#define DEFAULT_TOTALSIZE  1000
#define MAXSTR              200 

#define DEBUG(x)

double tim(void);
void gettime(disktime **, long, int *, int *);
void kill_signal (int);
int setup_net(int isserver, char *hostname, int port, int window_size, int *sock);
void *netread(void *arg);
void save_summary(disktime times[], char *filename, char *sumpath, 
		  int ntime);

#define PRINTSPEED(k, file) { \
  speed = times[k].size/(times[k].time-times[k-1].time)/1000/1000; /* MB/s */\
  fprintf(file, "%6d  %4lld  %9.3f   %6.2f Mbps ( %ld  %6.3f )\n", \
 	  k, totalsize/1000/1000, times[k].time-firsttime, \
 	  speed*8, times[k].size, (times[k].time-times[k-1].time)); \
}

volatile int time_to_quit = 0;
volatile int sig_received = 0;

/* Globals needed by network threads */
int  nfullbuf;
unsigned long long totalsize;
int bwrote;        /* Amount of net data written between updates */
int dodisk = 0;    /* Read/write to disk */
int bufsize;
int write_failure;

char *buf[NBUF];
ssize_t nbufsize[NBUF];
pthread_mutex_t bufmutex[NBUF];
pthread_mutex_t globalmutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t startwait_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  start_wait  = PTHREAD_COND_INITIALIZER;

int main (int argc, char * const argv[]) {
  int i, j, status, ofile, opt, tmp, nwrote, ntime, maxtimes, ibuf, jbuf;
  int sock, serversock, newfile, nupdate, done;
  char msg[MAXSTR+50], *fname;
  double firsttime, t1;
  float ftmp, speed;
  double tbehind;
  unsigned long long accum;
  pthread_t netthread;
  struct sockaddr_in client;    /* Client address */
  socklen_t client_len;
  disktime *times;

  unsigned long long currentfilesize, completed;
  
  int server = 1;
  int dodisk = 1;    /* Write to disk? */
  int debug = 0; 
  int summary = 0;       /* Save a summary of statistics */
  int oneshot = 0;       /* Quit after first receive */
  float timeupdate = 50; /* Update every 50 MB by default */
  int port = 52100;    
  int window_size = -1;	
  char hostname[MAXSTR+1] = ""; /* Host name to send data to */
  
  struct option options[] = {
    {"memory", 0, 0, 'm'},
    {"port", 1, 0, 'p'},
    {"host", 1, 0, 'H'},
    {"hostname", 1, 0, 'H'},
    {"window", 1, 0, 'w'},
    {"blocksize", 1, 0, 'b'},
    {"timeupdate", 1, 0, 't'},
    {"summary", 0, 0, 's'},
    {"oneshot", 0, 0, 'o'},
    {"debug", 0, 0, 'D'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  bufsize   = DEFAULT_BUFSIZE * 1000*1000;
  ofile = -1;
  i = -1;
  write_failure = 0;

  /* Read command line options */
  while (1) {
    opt = getopt_long_only(argc, argv, "DhH:", 
			   options, NULL);
    if (opt==EOF) break;

    switch (opt) {
      
    case 'b':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1)
	fprintf(stderr, "Bad blocksize option %s\n", optarg);
      else 
	bufsize = ftmp * 1000*1000;
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
      server = 0;
      break;

    case 'o':
      oneshot = 1;
      break;

    case 'D':
      debug = 1;
      break;

    case 's':
      summary = 1;
      break;
      
    case 'm':
      dodisk = 0;
      break;
      
    case 'h':
      printf("Usage: vsib_recv [options]\n");
      printf("  -p/-port <PORT   >    Port number for transfer\n");
      printf("  -b/-blocksize <SIZE>  Read/write block size\n");
      printf("  -w/-window <SIZE>     Network window size (kB)\n");
      printf("  -t/-time              Number of blocks (-b) to average timing statistics\n");
      printf("  -H/-host              Act as client and connect to remove server\n");
      printf("  -o/-oneshot           Quit after initial receive\n");
      printf("  -s/summary            Save a summary of network statistics\n");
      printf("  -memory               Don't write data to disk\n");
      printf("  -h/-help              This list\n");
      return(1);
    break;
    
    case '?':
    default:
      break;
    }
  }

  nupdate = timeupdate/(bufsize/1e6);
  if (nupdate==0) nupdate = 1;

  // Initialise buffers and syncronisation variables
  for (i=0; i<NBUF; i++) {
    buf[i] = malloc(bufsize);
    if (buf[i]==NULL) {
      sprintf(msg, "Trying to allocate %d MB", bufsize/1000/1000);
      perror(msg);
      return(1);
    }
  }
  
  maxtimes = 500;
  times = malloc(sizeof(disktime)*maxtimes);
  if (times==NULL) {
    perror(NULL);
    return(1);
  }
  
  ntime = 0;
  accum = 0;
  firsttime = -1;

  status = setup_net(server, hostname, port, window_size, &serversock);
  if (status) return(1);


  while (!time_to_quit) { // Loop over multiple connections

    if (server) {
      printf("Waiting for connection\n");

      /* Accept connection */
      client_len = sizeof(client);
      sock = accept(serversock, (struct sockaddr *)&client, &client_len);
      if (sock==-1) {
	perror("Error connecting to client");
	close(serversock);
	return(1);
      }
      
      printf("Got a connection from %s\n",inet_ntoa(client.sin_addr));
    } else {
      sock = serversock; 
    }
    
    /* Install a ^C catcher */
    signal (SIGINT, kill_signal);
  
    // Initialise syncronisation variables
    for (i=0; i<NBUF; i++) {
      nbufsize[i] = 0;
      pthread_mutex_init(&bufmutex[i], NULL);
    }
    nfullbuf=0;


    totalsize=0;
    
    t1 = 0;
    
    bwrote = 0;
    gettime(&times, 0, &ntime, &maxtimes);
    firsttime = times[0].time;
    ibuf = 0;
    jbuf = NBUF-1;
    j = 0;

    // Lock the last mutex
    DEBUG(printf("MAIN:   Locked %d for thread\n", jbuf));
    pthread_mutex_lock( &bufmutex[jbuf] );
    
    /* Need to start netread thread */
    status = pthread_create( &netthread, NULL, netread, (void *)&sock);
    if (status) {
      printf("Failed to start net read thread: %d\n", status);
      exit(1);    
    }
    
    /* Wait on netread to lock the first buffer and unlock this one */
    DEBUG(printf("MAIN:   Wait on lock for buf %d\n", jbuf));
    pthread_mutex_lock( &bufmutex[jbuf] );
    DEBUG(printf("MAIN:   Got it\n"));
    
    /* Loop over files */
    done = 0;
    while (!done) {
      
      t1 = tim(); /* So we can time average write per file */
      tbehind = 0;
      
      /* Loop over buffers */
      completed = 0;
      newfile = 1;
      i = 0;
      while (newfile || completed < currentfilesize) {
	
	// Wait until buffer ibuf has some data for us 
	DEBUG(printf("MAIN:   Try and get lock on %d\n", ibuf));
	pthread_mutex_lock( &bufmutex[ibuf] );
	DEBUG(printf("MAIN:   Got it\n"));
	
	// Unlock the previous lock
	DEBUG(printf("MAIN:   Release lock %d\n", jbuf));
	pthread_mutex_unlock( &bufmutex[jbuf] );
	
	pthread_mutex_lock(&globalmutex);
	DEBUG(printf("MAIN:    Nfullbuf = %d\n", nfullbuf));
	if (nfullbuf==0) {
	  DEBUG(printf("MAIN:    Quitting\n"));
	  pthread_mutex_unlock(&globalmutex);
	  done = 1;
	  break;
	}
	pthread_mutex_unlock(&globalmutex);
	
	DEBUG(printf("MAIN: nbufsize = %d\n", nbufsize[ibuf]));
	
	if (newfile) { // Need to read filename
	  
	  newfile = 0;
	  
	  memcpy(&currentfilesize,  buf[ibuf], sizeof(long long));
	  
	  fname = buf[ibuf]+sizeof(long long);
	  
	  if (dodisk) {
	    /* Open the output file if necessary */
	    
	    // File name contained in buffer
	    ofile = open(fname, OPENOPTIONS,S_IRWXU|S_IRWXG|S_IRWXO); 
	    
	    if (ofile==-1) {
	      sprintf(msg, "Failed to open output file (%s)", fname);
	      perror(msg);
	      
	      if (summary)
		save_summary(times, "Transfer.summary", "", ntime);
	      return(1);
	    }
	    pthread_mutex_lock(&globalmutex);
	    
	    if (currentfilesize<1e6) {
	      printf("Writing %3lld KB to %s\n", currentfilesize/1000, 
		     fname);
	    } else if (currentfilesize>1e9) {
	      printf("Writing %.3f GB to %s\n", currentfilesize/1e9, 
		     fname);
	    } else {
	      printf("Writing %3lld MB to %s\n", currentfilesize/1000/1000, 
		     fname);
	    }
	    
	    pthread_mutex_unlock(&globalmutex);
	  } else {

	    if (currentfilesize<1e6) {
	      printf("Receiving %3lld KB from %s\n", currentfilesize/1000, 
		     fname);
	    } else if (currentfilesize>1e9) {
	      printf("Receiving %.3f GB from %s\n", currentfilesize/1e9, 
		     fname);
	    } else {
	      printf("Receiving %3lld MB from %s\n", currentfilesize/1000/1000, 
		     fname);
	    }
	  }
	  
	} else if (dodisk && nbufsize[ibuf]>0) {
	  
	  nwrote = write(ofile, buf[ibuf], nbufsize[ibuf]);
	  if (nwrote==-1) {
	    perror("Error writing outfile");
	    close(ofile);
	    //free(mem);
	    write_failure = 1;
	    pthread_mutex_unlock( &bufmutex[ibuf] );
	    pthread_exit(NULL);
	  } else if (nwrote!=nbufsize[ibuf]) {
	    fprintf(stderr, "Warning: Did not write all bytes! (%d/%d)\n",
		    nwrote, (int)nbufsize[ibuf]);
	  }
	  completed += nwrote;
	  
	} else { // if dodisk
	  completed += bufsize;
	}
	
	pthread_mutex_lock(&globalmutex);
	nfullbuf--;
	pthread_mutex_unlock(&globalmutex);
	
	if (!((i+1) % nupdate)) {
	  gettime(&times, bwrote, &ntime, &maxtimes);
	  bwrote = 0;
	  PRINTSPEED(ntime-1, stdout);
	}
	
	DEBUG(printf("MAIN: ibuf++\n"));
	jbuf = ibuf;
	ibuf = (ibuf+1)%NBUF;
	
	pthread_mutex_lock(&globalmutex);
	DEBUG(printf("MAIN:    (Nfullbuf = %d)\n", nfullbuf));
	pthread_mutex_unlock(&globalmutex);
	
	
	if (time_to_quit) break;
	i++;
      } /* Loop over buffers */

      if (bwrote>0) {
	gettime(&times, bwrote, &ntime, &maxtimes);
	bwrote = 0;
	PRINTSPEED(ntime-1, stdout);
      }
      
      if (dodisk) {
	
	/* Close the output file */
	if (!newfile) {
	  status = close(ofile);
	  if (status!=0) {
	    perror("Error closing output file");
	  }
	}
      }
      j++;

      if (time_to_quit || write_failure) break;

    } // Loop over disks

    status = close(sock);
    if (status!=0) {
      perror("Error closing socket");
    }

    status = pthread_join(netthread, NULL);
    if (status) {
      printf("ERROR: Return code from pthread_join is %d\n", status);
    }

    printf("Re-enabling signal\n");
    signal (SIGINT, SIG_DFL);
    if (time_to_quit) {
      raise (sig_received);
      signal (sig_received, SIG_DFL);
    }
    if (!server || oneshot) time_to_quit = 1;
  }

  speed = totalsize/(times[ntime-1].time-firsttime)/1000/1000;
  printf("  \nRate = %.2f Mbps/s (%.1f sec)\n\n", speed*8, 
	 times[ntime-1].time-firsttime);
  
  if (summary)
    save_summary(times, "transfer.summary", "", ntime);

  for (i=0; i<NBUF; i++) {
    if(buf[i]!=NULL) free(buf[i]);
    buf[i] = NULL;
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

void gettime(disktime **times, long size, int *ntimes, int *maxtimes) {
  disktime *newtimes;
  if (*ntimes>=*maxtimes) {
    /* Reallocates some more space */
    *maxtimes +=1000;
    newtimes = realloc(*times, sizeof(disktime)* *maxtimes);

    if (newtimes==NULL) {
      fprintf(stderr, "Realloc failed!\n");
      perror(NULL);
      exit(1);
    }
    *times = newtimes;
  }

  (*times)[*ntimes].time = tim();
  (*times)[*ntimes].size = size;
  (*ntimes)++;

  return;
}

int setup_net(int isserver, char *hostname, int port, int window_size,
	      int *sock) {
  int status;
  unsigned long ip_addr;
  socklen_t  winlen;
  struct hostent     *hostptr;
  struct sockaddr_in server;    /* Socket address */

  /* Open a server connection for reading */

  /* Initialise server's address */
  memset((char *)&server,0,sizeof(server));
  server.sin_family = AF_INET;
  server.sin_port = htons((unsigned short)port); /* Which port number to use */

  /* Create the initial socket */
  *sock = socket(AF_INET,SOCK_STREAM,0); 
  if (*sock==-1) {
    perror("Error creating socket");
    return(1);
  }

  if (window_size>0) {
    status = setsockopt(*sock, SOL_SOCKET, SO_SNDBUF,
			(char *) &window_size, sizeof(window_size));
    if (status!=0) {
      perror("Error setting socket send buffer");
      close(*sock);
      return(1);
    } 

    status = setsockopt(*sock, SOL_SOCKET, SO_RCVBUF,
			(char *) &window_size, sizeof(window_size));
    
    if (status!=0) {
      perror("Error setting socket receive buffer");
      close(*sock);
      return(1);
    }

    /* Check what the window size actually was set to */
    winlen = sizeof(window_size);
    status = getsockopt(*sock, SOL_SOCKET, SO_SNDBUF,
			(char *) &window_size, &winlen);
    if (status!=0) {
      close(*sock);
      perror("Getting socket options");
      return(1);
    }
    printf("Sending buffersize set to %d Kbytes\n", window_size/1024);
  }

  if (isserver) {

    /* Initialise server's address */
    server.sin_addr.s_addr = htonl(INADDR_ANY); /* Anyone can connect */
  
    status = bind(*sock, (struct sockaddr *)&server, sizeof(server));
    if (status!=0) {
      perror("Error binding socket");
      close(*sock);
      return(1);
    } 
  
    /* We are willing to receive conections, using a maximum
       back log of 1 */
    status = listen(*sock,1);
    if (status!=0) {
      perror("Error binding socket");
      close(*sock);
      return(1);
    }

  } else {  // Acting as client

    hostptr = gethostbyname(hostname);
    if (hostptr==NULL) {
      fprintf(stderr,"Failed to look up hostname %s\n", hostname);
      close(*sock);
      return(1);
    }

    memcpy(&ip_addr, (char *)hostptr->h_addr, sizeof(ip_addr));
    server.sin_addr.s_addr = ip_addr;

    printf("Connecting to %s\n",inet_ntoa(server.sin_addr));

    status = connect(*sock, (struct sockaddr *) &server, sizeof(server));
    if (status!=0) {
      perror("Failed to connect to server");
      close(*sock);
      return(1);
    }
  }

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

void *netread (void *arg) {
  short fnamesize=0;
  int sock;
  int ibuf = 0;
  int jbuf = NBUF-1;
  int newfile = 1;
  ssize_t nr;
  char *poff;
  unsigned long long bytesremaining = 0;
  size_t ntoread;

  sock = *(int*)arg;
  
  while (1) {

    if (time_to_quit || write_failure) {
      pthread_mutex_unlock( &bufmutex[jbuf] );
      pthread_exit(NULL);
    }

    if (!newfile || fnamesize==0) {
      pthread_mutex_lock( &bufmutex[ibuf] );
      pthread_mutex_unlock( &bufmutex[jbuf] );
    }
    nbufsize[ibuf] = 0;
    poff = buf[ibuf];

    if (newfile) {
      /* Data is prefixed by a header consting of:
	    8 byte file size (does not include these header values)
	    2 byte filename size (including terminating nul)
	    n bytes filename
         Then number of bytes of data given by this header
      */

      if (fnamesize==0) { // Have not read filesize and file name yet
	ntoread = sizeof(long long) + sizeof(short);
      } else {            // Need to read filename
	ntoread = fnamesize;

	poff += sizeof(long long); // Retain filesize header
      }
    } else {
      if (bytesremaining>bufsize) 
	ntoread = bufsize;
      else
	ntoread = bytesremaining;
    }

    while (nbufsize[ibuf]<ntoread) {
    
      nr = recv(sock,poff,ntoread-nbufsize[ibuf],0);
    	  
      if (nr==-1) {
	if (errno == EINTR) continue;
	perror("Error reading socket");
	pthread_mutex_unlock( &bufmutex[ibuf] );
	pthread_exit(NULL);

      } else if (nr==0) {
	/* Assume for now this means the socket was closed */
	pthread_mutex_lock(&globalmutex);
	if (nbufsize[ibuf]>0) nfullbuf++;
	pthread_mutex_unlock(&globalmutex);
	pthread_mutex_unlock(&bufmutex[ibuf]);
	pthread_exit(NULL);

      } else {
	nbufsize[ibuf] += nr;
	poff += nr;
	if (!newfile) {
	  pthread_mutex_lock(&globalmutex);
	  totalsize += nr;
	  bwrote += nr;
	  pthread_mutex_unlock(&globalmutex);
	  bytesremaining -= nr;
	}
      }
    }

    if (newfile) {
      if (fnamesize==0) { // Just read filesize and filename size

	memcpy(&bytesremaining,  buf[ibuf], sizeof(long long));

	memcpy(&fnamesize,  buf[ibuf]+sizeof(long long), sizeof(short));

	if (fnamesize>bufsize-sizeof(long long)) {
	  fprintf(stderr, "Passed filename too long\n");
	  pthread_mutex_unlock( &bufmutex[ibuf] );
	  pthread_exit(NULL);
	} else if (fnamesize<=0) {
	  fprintf(stderr, "Received bad filename size %d\n", fnamesize);
	  pthread_mutex_unlock( &bufmutex[ibuf] );
	  pthread_exit(NULL);
	}
      } else {
	newfile = 0;

	poff = buf[ibuf]+ sizeof(long long);
      }
    }

    if (!newfile) {  // Reuse this buffer if receiving file name
      pthread_mutex_lock(&globalmutex);
      nfullbuf++;
      pthread_mutex_unlock(&globalmutex);

      jbuf = ibuf;
      ibuf  = (ibuf+1)%NBUF;
    }

    if (!newfile && bytesremaining<=0 ) { // Received all of file
      newfile = 1;
      bytesremaining = 0;
      fnamesize = 0;
    }
  }
}

void save_summary(disktime times[], char *filename, char *sumpath, 
		  int ntime) {
  int i, status;
  float speed;
  double firsttime;
  unsigned long long totalsize;
  char newfilename[MAXSTR+1], msg[MAXSTR+50], *cptr;
  FILE *fsum;

  if (strlen(sumpath)>0) {
    /* First we need to strip any path into from filename if present*/
    if ((cptr=rindex(filename, '/'))==NULL) {
      cptr = filename;
    } else {
      cptr++; /* What happens if file name ends with a '/'? */
    }
    strcpy(newfilename, sumpath);
    strcat(newfilename, "/");
    strcat(newfilename, cptr);
  } else {
    strcpy(newfilename, filename);
  }

  fsum = fopen(newfilename, "w");

  if (fsum==NULL) {
    sprintf(msg, "Failed to open summary file (%s)", newfilename);
    perror(msg);

    if (strcmp(sumpath, "/tmp")!=0) {
      fprintf(stderr, "Trying to save on /tmp\n");
      save_summary(times, filename, "/tmp", ntime);
    }
    return;
  }

  printf("Saving summary file as %s\n", newfilename);
  
  firsttime = times[0].time;
  totalsize = 0;

  fprintf(fsum, "Count      Total Time (sec)        Nwrote (bytes)\n");
  fprintf(fsum, "     Filesize (MB)       Speed            Delta Time (sec)\n");

  for (i=1; i<ntime; i++) {
    totalsize += times[i].size;
    PRINTSPEED(i, fsum);
  }
 
  status = fclose(fsum);
  if (status!=0) {
    if (errno==ENOSPC) {
      /* This filesystem is full. Try and write to /tmp */

      if (strcmp(sumpath, "/tmp")==0) {
	fprintf(stderr, "/tmp full also! Giving up\n");
	return;
      } else {
	fprintf(stderr, "Filesytem full, could not save summary file. "
		"Trying to save on /tmp\n");
      }
      save_summary(times, filename, "/tmp", ntime);

    } else {
      perror("Error closing summary file");
    }
  }
}
