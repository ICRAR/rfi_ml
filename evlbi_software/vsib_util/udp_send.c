#ifdef __APPLE__
#define OSX
#define SENDOPTIONS  0
#else
#define LINUX
#define _LARGEFILE_SOURCE 
#define _LARGEFILE64_SOURCE
#define SENDOPTIONS MSG_NOSIGNAL
#endif

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <limits.h>
#include <netinet/in.h>
#include <netdb.h>
#include <sys/types.h>  
#include <sys/socket.h>  
#include <sys/time.h>  
#include <string.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>


#define MAXSTR              200 
#define DEFAULT_PACKETSIZE 1500
#define DEFAULT_NPACKET  100000
#define DEFAULT_PORT      52100

#define DEBUG(x) 

int setup_net(char *hostname, unsigned short port, int *sock);
void my_usleep(u_int64_t usec);

int main (int argc, char * const argv[]) {
  unsigned int port = DEFAULT_PORT;
  unsigned int packetsize = DEFAULT_PACKETSIZE;
  int npacket = DEFAULT_NPACKET;
  int isclient = 0; /* Client or server */
  char *buf, *ptr;
  char hostname[MAXSTR+1] = ""; /* Host name to send data to */
  unsigned long long *sequence;
  int tmp, opt, status, sock, i;
  int ipd = 0;
  ssize_t nwrote;
  char msg[MAXSTR];
  
  struct option options[] = {
    {"packetsize", 1, 0, 's'},
    {"size", 1, 0, 's'},
    {"hostname", 1, 0, 'h'},
    {"npacket", 1, 0, 'n'},
    {"packet", 1, 0, 'n'},
    {"ipd", 1, 0, 'i'},
    {"port", 1, 0, 'p'},
    {"help", 0, 0, 'H'},
    {0, 0, 0, 0}
  };

  srand(time(NULL));

  if (sizeof(unsigned long long)!=8) {
    fprintf(stderr, "Error: Program assumes long long is 8 bytes\n");
    exit(1);
  }

  while (1) {
    opt = getopt_long_only(argc, argv, "p:s:n:h:cdH", 
			   options, NULL);
    if (opt==EOF) break;

    switch (opt) {
      
    case 's':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1 || tmp<=0 || tmp>USHRT_MAX)
	fprintf(stderr, "Bad packetsize option %s\n", optarg);
      else 
	packetsize = tmp;
     break;

    case 'p':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1 || tmp<=0 || tmp>USHRT_MAX)
	fprintf(stderr, "Bad port option %s\n", optarg);
      else 
	port = tmp;
     break;

    case 'n':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1 || tmp<=0)
	fprintf(stderr, "Bad npacket option %s\n", optarg);
      else 
	npacket = tmp;
      break;

    case 'i':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1 || tmp<=0)
	fprintf(stderr, "Bad ipd option %s\n", optarg);
      else 
	ipd = tmp;
      break;

    case 'h':
      if (strlen(optarg)>MAXSTR) {
	fprintf(stderr, "Hostname too long\n");
	return(1);
      }
      strcpy(hostname, optarg);
      isclient = 1;
      break;

    case 'H':
      printf("Usage: diskspeed [options]\n");
      printf("  -s/-packetsize <SIZE>   UDP block size\n");
      printf("  -n/-npacket <N>         Number of packets to sent\n");
      printf("  -p/-port <PORT>        Port to use\n");
      printf("  -h/-hostname <HOSTNAME> Remote host name\n");
      printf("  -h/-help                This list\n");
      return(1);
    break;
    
    case '?':
    default:
      break;
    }
  }

  if (strlen(hostname) == 0) {
    strcpy(hostname, "localhost");
  }

  packetsize -= 20+2*4;

  buf = malloc(packetsize);
  if (buf==NULL) {
    sprintf(msg, "Trying to allocate %d bytes", packetsize);
    perror(msg);
    return(1);
  }

  /* Fill the the buffer with random bits */
  ptr = buf;
  for (i=0; i<packetsize; i++) {
    *ptr = (unsigned char)(256.0*rand()/(RAND_MAX+1.0));
    ptr++;
  }

  status = setup_net(hostname, port, &sock);
    
  if (status)  exit(1);

  sequence = (unsigned long long*)buf;

  *sequence = 0;

  while (npacket) {
    /* Send n packets to server */

    //printf("Sending packet %llu\n", *sequence+1);

    nwrote = send(sock, buf,  packetsize, MSG_EOR);

    if (nwrote==-1) {
      perror("Sending packet");
      exit(1);
    } else if (nwrote==0) {
      fprintf(stderr, "Warning: Did not write any bytes!\n");
      exit(1);

    } else if (nwrote!=packetsize) {
      fprintf(stderr, "Warning: Did not write all bytes!\n");
      exit(1);
    }

    (*sequence)++;
    npacket--;

    if (ipd)  my_usleep(ipd);
  } //  while npacket

  free(buf);

  return(0);
}
  
int setup_net(char *hostname, unsigned short port, int *sock) {
  int status;
  unsigned long ip_addr;
  struct hostent     *hostptr;
  struct sockaddr_in server; 

  /* Initialise server's address */
  memset((char *)&server,0,sizeof(server));
  server.sin_family = AF_INET;
  server.sin_port = htons(port); /* Which port number to use */

  /* Create socket */
  *sock = socket(AF_INET,SOCK_DGRAM, IPPROTO_UDP); 
  if (*sock==-1) {
    perror("Error creating socket");
    return(1);
  }

  /* Look up hostname */
  hostptr = gethostbyname(hostname);
  if (hostptr==NULL) {
    fprintf(stderr,"Failed to look up hostname %s\n", hostname);
    return(1);
  }
  memcpy(&ip_addr, (char *)hostptr->h_addr, sizeof(ip_addr));

  printf("IP ADDR = %lu for %s\n", ip_addr, hostname);
  server.sin_addr.s_addr = ip_addr;

  status = connect(*sock, (struct sockaddr *) &server, sizeof(server));
  if (status!=0) {
    perror("Failed to connect to server");
    return(1);
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
  
void my_usleep(u_int64_t usec) {
  double now, till;
  int n;
  /* The current time */
  now = tim();

  /* The time we are sleeping till */
  till = now+usec/1.0e6;

  /* and spin for the rest of the time */
  n = 0;
  while (now<till) {
    now = tim();
    n++;
  }
  
}
