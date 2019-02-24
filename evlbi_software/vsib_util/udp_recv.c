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
#include <unistd.h>
#include <netinet/in.h>
#include <sys/types.h>  
#include <sys/socket.h>  
#include <sys/time.h>  
#include <string.h>
#include <signal.h>
#include <getopt.h>
#include <time.h>
#include <netdb.h>
#include <limits.h>

#define MAXSTR              200 
#define MAXPACKETSIZE       65536
#define DEFAULT_PORT        52100
#define DEFAULT_UPDATETIME  1.0
#define UPDATE              20

#define DEBUG(x) 

double tim(void);
void alarm_signal (int sig);

int setup_net(unsigned short port, int *sock);

volatile int time_to_quit = 0;
volatile int sig_received = 0;

#define INIT_STATS() { \
  npacket = 0; \
  maxpkt_size = 0; \
  minpkt_size = ULONG_MAX; \
  sum_pkts = 0; \
  pkt_drop = 0; \
  pkt_oo = 0; \
}

/* Globals needed by alarm handler */

unsigned long minpkt_size, maxpkt_size, pkt_drop, pkt_oo;
unsigned long long sum_pkts, pkt_head, npacket;
double t1, t2;

int lines = 0;

int main (int argc, char * const argv[]) {
  unsigned int port = DEFAULT_PORT;
  char *buf;
  int tmp, opt, status, sock;
  unsigned long long sequence;
  ssize_t nread;
  char msg[MAXSTR];
  float updatetime, ftmp;
  struct itimerval timeval;
  sigset_t set;
  struct msghdr udpmsg;
  struct iovec iov[2];
  
  struct option options[] = {
    {"port", 1, 0, 'p'},
    {"time", 1, 0, 't'},
    {"updatetime", 1, 0, 't'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  srand(time(NULL));

  if (sizeof(unsigned long long)!=8) {
    fprintf(stderr, "Error: Program assumes long long is 8 bytes\n");
    exit(1);
  }

  updatetime = DEFAULT_UPDATETIME;

  while (1) {
    opt = getopt_long_only(argc, argv, "p:t:h", options, NULL);
    if (opt==EOF) break;

    switch (opt) {
      
    case 'p':
      status = sscanf(optarg, "%d", &tmp);
      if (status!=1 || tmp<=0 || tmp>USHRT_MAX)
	fprintf(stderr, "Bad port option %s\n", optarg);
      else 
	port = tmp;
     break;

    case 't':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1 || ftmp<=0.0)
	fprintf(stderr, "Bad updatetime option %s\n", optarg);
      else 
	updatetime = ftmp;
     break;

    case 'h':
      printf("Usage: udp_recv [options]\n");
      printf("  -p/-port <PORT>        Port to use\n");
      printf("  -h/-help               This list\n");
      return(1);
    break;
    
    case '?':
    default:
      break;
    }
  }

  buf = malloc(MAXPACKETSIZE);
  if (buf==NULL) {
    sprintf(msg, "Trying to allocate %d bytes", MAXPACKETSIZE);
    perror(msg);
    return(1);
  }

  status = setup_net(port, &sock);
    
  if (status)  exit(1);

  INIT_STATS();
  pkt_head = 0;

  /* Install an alarm catcher */
  signal(SIGALRM, alarm_signal);
  sigemptyset(&set);
  sigaddset(&set, SIGALRM);

  timeval.it_interval.tv_sec = updatetime; 
  timeval.it_interval.tv_usec = 0;
  timeval.it_value.tv_sec = updatetime;
  timeval.it_value.tv_usec = 0;
  t1 = tim();
  setitimer(ITIMER_REAL, &timeval, NULL);

  memset(&udpmsg, 0, sizeof(udpmsg));
  udpmsg.msg_iov        = &iov[0];
  udpmsg.msg_iovlen     = 2;
  iov[0].iov_base = &sequence;
  iov[0].iov_len = sizeof(sequence);
  iov[1].iov_base = buf;
  iov[1].iov_len = MAXPACKETSIZE;

  while(1) {
    //nread = recv(sock, buf, MAXPACKETSIZE, MSG_WAITALL);
    nread = recvmsg(sock,&udpmsg,MSG_WAITALL);
    
    
    if (nread==-1) {
      perror("Receiving packet");
      exit(1);
    } else if (nread==0) {
      fprintf(stderr, "Warning: Did not read any bytes!\n");
      exit(1);
    } else if (nread==MAXPACKETSIZE) {
      fprintf(stderr, "Warning: Packetsize larger than expected!\n");
      exit(1);
    } else if (nread<sizeof(long long)) {
      fprintf(stderr, "Warning: Too few bytes read!\n");
      exit(1);
    }

    // Block alarm signal while we are updating these values
    status = sigprocmask(SIG_BLOCK, &set, NULL);
    if (status) {
      perror(": Trying to block SIGALRM\n");
      exit(1);
    }

    npacket++;
    if (nread>maxpkt_size) maxpkt_size = nread;
    if (nread<minpkt_size) minpkt_size = nread;
    sum_pkts += nread;

    if (sequence==0) {
      pkt_head = 0;
    } else if (sequence<pkt_head) {  // This could be duplicates also
      pkt_oo++;
      pkt_drop--;
    } else {
      if (sequence>pkt_head+1) {
	pkt_drop += sequence-pkt_head-1;
      }
      pkt_head = sequence;
    }

    // Unblock the signal again
    status = sigprocmask(SIG_UNBLOCK, &set, NULL);
    if (status) {
      perror(": Trying to block SIGALRM\n");
      exit(1);
    }

  }

  free(buf);

  return(0);
}
  
int setup_net(unsigned short port, int *sock) {
  int status;
  struct sockaddr_in server; 

  /* Initialise server's address */
  memset((char *)&server,0,sizeof(server));
  server.sin_family = AF_INET;
  server.sin_port = htons(port); 

  /* Create socket */
  *sock = socket(AF_INET,SOCK_DGRAM, IPPROTO_UDP); 
  if (*sock==-1) {
    perror("Error creating socket");
    return(1);
  }

  int udpbufbytes = 10*1024*1024;
  status = setsockopt(*sock, SOL_SOCKET, SO_RCVBUF,
		      (char *) &udpbufbytes, sizeof(udpbufbytes));
    
  if (status!=0) {
    fprintf(stderr, "Warning: Could not set socket RCVBUF\n");
  } 

  server.sin_addr.s_addr = htonl(INADDR_ANY); /* Anyone can connect */

  status = bind(*sock, (struct sockaddr *)&server, sizeof(server));
  if (status!=0) {
    perror("Error binding socket");
    close(*sock);
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

void alarm_signal (int sig) {
  long avgpkt;
  float delta, rate, percent;
  t2 = tim();
  delta = t2-t1;

  if (npacket==0) {
    avgpkt = 0;
    minpkt_size = 0;
  } else 
    avgpkt = sum_pkts/npacket;
  rate = sum_pkts*8/delta/1e6;

  if (lines % UPDATE ==0) {
    printf("  npkt   min  max  avg sec Rate Mbps  drop  %%    oo\n");
  }
  lines++;

  if (npacket==0)
    percent = 0; 
  else 
    percent = (double)pkt_drop/(npacket+pkt_drop)*100;

  printf("%6llu  %4lu %4lu %4ld %3.1f  %7.3f %5lu %5.2f %3lu\n", 
	 npacket, minpkt_size, maxpkt_size, avgpkt, delta, rate, 
	 pkt_drop, percent, pkt_oo);	
  INIT_STATS();
  t1 = t2;
  return;
}  

