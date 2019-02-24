/* #define _GNU_SOURCE */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#define BUFSIZE 1000

static char *year_months[] =
  {
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
  };
static int n_year_months=12;

struct _last_status {
  time_t file_time;
  char   last_filename[BUFSIZE];
  int    last_block;
  int    last_BIGBUF_mbyte;
  int    last_BIGBUF_pct;
  int    last_PPS_OK;
  char   pps_error[BUFSIZE];
  char   statistics[BUFSIZE];
};

int keep_running;
struct _last_status last_status;
int status_waiting;
int threads_mode;

void stop_program(int sig);
void parse_line(char *textline);
void determine_filetime(char *filename);
void send_start_time(char *mesg);
void send_immediate_warning(char *mesg);
void send_recorder_status(void);
void server_send(char *message);
void clear_status(void);
void server_comms(char *message);

/* char *cfgets(char *buffer,int maxlength,int fdesc); */
/* int input_timeout(int filedes,unsigned int seconds); */

int main(int argc,char *argv[]){
  char input_line[BUFSIZE];
  FILE *input_file=NULL;

  /*
   This program monitors a file of output from the
   vsib_record command to check that the PPS signals
   are being received correctly by the recorder, and
   that the BIGBUF levels are adequate. Technically
   though, this program just gets the values of the
   BIGBUF level and PPS signal, and passes them via
   network socket to the recorder server.
  */

  /* the name of the file to monitor is passed to 
     this program as the first/only argument */
  if (argc!=2){
    printf("Usage:\n");
    printf("%s <filename>\n",argv[0]);
    printf("where <filename> is the file which has the output from the\n");
    printf("recorder.\n");
    exit(0);
  }

  /* Ctrl-C stops the program */
  (void)signal(SIGINT,stop_program);

  /* by default, we don't expect output from a thread recorder */
  threads_mode=0;

  /* open the file */
  if ((input_file=fopen(argv[1],"r"))==NULL){
    printf("Cannot open file %s!\n",argv[1]);
    exit(0);
  }

  /* keep running until Ctrl-C is pressed */
  keep_running=1;
  status_waiting=0; /* nothing read yet */
  while(keep_running){
    if ((fgets(input_line,BUFSIZE,input_file))!=NULL){
      /* we're not at the EOF */
/*       printf("%s",input_line); */
      parse_line(input_line);
    } else {
      /* EOF - no more text in the file */
      sleep(1); /* wait for a second */
    }
  }

  /* if we get here, Ctrl-C has been pressed */
  while ((fgets(input_line,BUFSIZE,input_file))!=NULL){
    parse_line(input_line);
  }
  fclose(input_file);
  exit(0);

}

void stop_program(int sig){
  /* this routine is called when Ctrl-C is pressed 
     (ie the SIGINT signal is received). it simply
     changes the global flag keep_running to false
     so the main while loop exits */
  keep_running=0;
}

void parse_line(char *textline){
  int blocknum,BIGBUF_pct;/* ,dayofyear,daydiff; */
  long unsigned int BIGBUF_mbytes;
  char *filename;
/*   time_t file_time; */
/*   struct tm time_file; */

  filename=(char *)malloc(BUFSIZE*sizeof(char));
  /*
    this routine takes a line from the vsib_record
    output file and works out what it says, before
    taking the appropriate action and calling the
    routine to send the data to the server
  */

  /*
    during normal operation, the recorder should
    produce output similar to the following:
    Sun Jan  7 23:15:19 UTC 2007
    at block = 0, opened file 'test_007_231520.lba'
    1PPS OKAY
    MBytes left in the BIGBUF 487 (100%)
    Chan 0:  38.27  61.73   0.00   0.00
    Chan 1:  38.27  61.73   0.00   0.00
    Chan 2:  49.33  50.54   0.07   0.06
    Chan 3:  49.33  50.54   0.07   0.06

    if the 1PPS signal has been missed, vsib_record
    will produce a message similar to:

    ERROR: 1PPS transition ABSENT (05) at 20070107:231527

    during normal threaded operation, the recorder should
    produce output similar to the following:
    1PPS OKAY
    MBytes left in the BIGBUF 487 (100%)
    Opened file 'test_At_080_031810.lba'
    DEBUG: Sending 320004096 bytes for test_At_080_031810.lba [23]
    Chan 0:   0.00  50.00  50.00   0.00
    Chan 1:   0.00  50.00  50.00   0.00
    Chan 2:   0.00  50.00  50.00   0.00
    Chan 3:   0.00  50.00  50.00   0.00

  */
  if (strncmp(textline,"at block =",10)==0){
/*     printf("%s",textline); */
/*     if (threads_mode==0){ */
/*       if (status_waiting){ */
/* 	send_recorder_status(); */
/*       } */
/*       clear_status(); */
/*     } */
    sscanf(textline,"at block = %d, opened file '%[^']'",
	   &blocknum,filename);
    last_status.last_block=blocknum;
    strcpy(last_status.last_filename,filename);
/*     printf("<%s>\n",(filename+=(strlen(filename)-14))); */
/*     filename+=(strlen(filename)-14); */
/*     time(&file_time); */
/*     gmtime_r(&file_time,&time_file); */
/*     sscanf(filename,"%3d_%2d%2d%2d.lba", */
/* 	   &dayofyear,&(time_file.tm_hour), */
/* 	   &(time_file.tm_min),&(time_file.tm_sec)); */
/*     last_status.file_time=mktime(&time_file); */
/*     daydiff=(time_file.tm_yday+1)-dayofyear; */
/*     time_file.tm_mday-=daydiff; */
/*     if (daydiff<0){ */
/*       time_file.tm_year--; */
/*     } */
/*     last_status.file_time=timegm(&time_file); */
    determine_filetime(filename+strlen(filename)-14);
/*     if (threads_mode==0){ */
/*       last_status.statistics[0]='\0'; */
/*       status_waiting=1; */
/*     } */
  } else if (strncmp(textline,"Opened file '",13)==0){
    sscanf(textline,"Opened file '%[^']'",filename);
    strcpy(last_status.last_filename,filename);
    last_status.last_block=0;
/*     filename+=(strlen(filename)-14); */
    determine_filetime(filename+strlen(filename)-14);
  } else if (strncmp(textline,"MBytes left in the BIGBUF",25)==0){
/*     printf("%s",textline); */
/*     if (threads_mode==1){ */
      if (status_waiting){
	send_recorder_status();
      }
      clear_status();
/*     } */
    sscanf(textline,"MBytes left in the BIGBUF %lu (%d)",
	   &(BIGBUF_mbytes),&(BIGBUF_pct));
    last_status.last_BIGBUF_mbyte=BIGBUF_mbytes;
    last_status.last_BIGBUF_pct=BIGBUF_pct;
/*     if (threads_mode==1){ */
      last_status.statistics[0]='\0';
      status_waiting=1;
/*     } */
  } else if (strncmp(textline,"1PPS OKAY",9)==0){
    last_status.last_PPS_OK=1;
  } else if (strncmp(textline,"ERROR: 1PPS transition ABSENT",29)==0){
    strcpy(last_status.pps_error,textline);
    last_status.last_PPS_OK=0;
    send_immediate_warning(textline);
  } else if (strncmp(textline,"Warning: Header time is offset from wall clock",
		     46)==0){
    send_immediate_warning(textline);
  } else if (strncmp(textline,"Chan ",5)==0){
    strcat(last_status.statistics,textline);
  } else if (strncmp(textline,"Start will occur at:",20)==0){
    send_start_time(textline);
  } else if (strncmp(textline,"**** THREAD",11)==0){
    threads_mode=1;
  }
}

void determine_filetime(char *filename) {
  time_t file_time;
  struct tm time_file;
  int dayofyear,daydiff;

  time(&file_time);
  gmtime_r(&file_time,&time_file);
  sscanf(filename,"%3d_%2d%2d%2d.lba",
	 &dayofyear,&(time_file.tm_hour),
	 &(time_file.tm_min),&(time_file.tm_sec));
  last_status.file_time=mktime(&time_file);
  daydiff=(time_file.tm_yday+1)-dayofyear;
  time_file.tm_mday-=daydiff;
  if (daydiff<0){
    time_file.tm_year--;
  }
  last_status.file_time=timegm(&time_file);

}

void send_start_time(char *mesg){
  char st_time[BUFSIZE],send_string[BUFSIZE];
  char st_month[BUFSIZE],*nl=NULL;
  int year,month,day,hour,minute,second,i;

  month = -1;

  sscanf(mesg,"Start will occur at: %[^|]",st_time);

  if ((nl=strstr(st_time,"\n"))!=NULL)
    nl[0]='\0';

  sscanf(st_time,"%4d %s %d %d:%d:%d UTC",
	 &year,st_month,&day,&hour,&minute,&second);
  for (i=0;i<n_year_months;i++){
    if (strncmp(st_month,year_months[i],
		strlen(year_months[i]))==0){
      month=i+1;
      break;
    }
  }

  snprintf(send_string,BUFSIZE,"<rcss>%4d%02d%02d_%02d%02d%02d</rcss>",
	   year,month,day,hour,minute,second);
  server_send(send_string);  
}

void send_immediate_warning(char *mesg){
  time_t time_now;
  struct tm now_time;
  char send_string[BUFSIZE],tmp[BUFSIZE];

  /*
    this routine sends a warning string back to the
    recorder server via the normal port 50080 network
    socket, except we use 127.0.0.1 as the destination
    address, ie. this program must be running on
    the same computer that the recorder_server
    process is running on.
    the string looks like
    <rcsw>ddmmyyyyhhmmss|warning text</rcsw>
    where
     ddmmyyyyhhmmss is the time the last file was opened
     warning text is the warning message
  */

  snprintf(send_string,BUFSIZE,"<rcsw>");
  time(&time_now);
  gmtime_r(&time_now,&now_time);
  strcpy(tmp,send_string);
  snprintf(send_string,BUFSIZE,"%s%02d%02d%04d%02d%02d%02d|",
           tmp,now_time.tm_mday,now_time.tm_mon+1,
           now_time.tm_year+1900,now_time.tm_hour,
           now_time.tm_min,now_time.tm_sec);
  strcpy(tmp,send_string);
  snprintf(send_string,BUFSIZE,"%s%s",
           tmp,mesg);
  strncat(send_string,"</rcsw>",7);
/*   printf("%s\n",send_string); */
  server_send(send_string);
}

void send_recorder_status(void){
  struct tm file_time;
  char send_string[BUFSIZE],tmp[BUFSIZE];

  /*
    this routine sends a status string back to the
    recorder server via the normal port 50080 network
    socket, except we use 127.0.0.1 as the destination
    address, ie. this program must be running on
    the same computer that the recorder_server
    process is running on.
    the string looks like
    <rcst>ddmmyyyyhhmmss|filename|block|BIGBUF|pct|PPS|sampler_stats</rcst>
    where
     ddmmyyyyhhmmss is the time the last file was opened
     filename is the name of the last opened file
     block is the first block of the last opened file
     BIGBUF is the number of MB left in the BIGBUF
     pct is the percentage free memory in the BIGBUF
     PPS is either OK for no error, or the error message
     sampler_stats is the sampler statistics string output by vsib_record
  */

  snprintf(send_string,BUFSIZE,"<rcst>");
  gmtime_r(&last_status.file_time,&file_time);
  strcpy(tmp,send_string);
  snprintf(send_string,BUFSIZE,"%s%02d%02d%04d%02d%02d%02d|",
	   tmp,file_time.tm_mday,file_time.tm_mon+1,
	   file_time.tm_year+1900,file_time.tm_hour,
	   file_time.tm_min,file_time.tm_sec);
  strcpy(tmp,send_string);
  snprintf(send_string,BUFSIZE,"%s%s|%d|%d|%d|",
	   tmp,last_status.last_filename,last_status.last_block,
	   last_status.last_BIGBUF_mbyte,last_status.last_BIGBUF_pct);
  if (last_status.last_PPS_OK)
    strncat(send_string,"OK",2);
  else
    strncat(send_string,last_status.pps_error,
	    strlen(last_status.pps_error));
  strncat(send_string,"|",1);
  strncat(send_string,last_status.statistics,
	  strlen(last_status.statistics));
  strncat(send_string,"</rcst>",7);
/*   printf("%s\n",send_string); */
  server_send(send_string);
}

void server_send(char *message){
  /*
    this routine calls the server communication
    routine and then blanks the last_status
    structure so it can be used again the next
    time we have output from vsib_record
  */
  server_comms(message);
/*   printf("%s\n",message); */
}

void clear_status(void){
/*   last_status.file_time=-1; */
/*   last_status.last_filename[0]='\0'; */
/*   last_status.last_block=-1; */
  last_status.last_BIGBUF_mbyte=-1;
  last_status.last_BIGBUF_pct=-1;
  last_status.last_PPS_OK=-1;
  last_status.pps_error[0]='\0';
}

void server_comms(char *message){
  int sock;
  struct sockaddr_in mesgserver;
  char buffer[BUFSIZE],failmsg[BUFSIZE];
  unsigned int mesglen;
  int received = 0;

  /* Create the TCP socket */
  if ((sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0) {
    printf("Failed to create socket\n");
    return;
  }

  /* Construct the server sockaddr_in structure */
  memset(&mesgserver, 0, sizeof(mesgserver));       /* Clear struct */
  mesgserver.sin_family = AF_INET;                  /* Internet/IP */
  mesgserver.sin_addr.s_addr = inet_addr("127.0.0.1");  /* IP address */
  mesgserver.sin_port = htons(50080);       /* server port */
  /* Establish connection */
  if (connect(sock,
	      (struct sockaddr *) &mesgserver,
	      sizeof(mesgserver)) < 0) {
    printf("Failed to connect with server\n");
    close(sock);
    return;
  }

  /* Send the word to the server */
  mesglen = strlen(message);
  if (send(sock, message, mesglen, 0) != mesglen) {
    printf("Mismatch in number of sent bytes\n");
    close(sock);
  }

  /* Receive the confirmation back from the server */
  if ((received=recv(sock,buffer,BUFSIZE-1,0))<1){
    printf("Failed to receive confirmation message from server\n");
    close(sock);
    return;
  } else {
    if (strcmp(buffer,"<succ />")==0){
    } else {
      sscanf(buffer,"<fail>%[^<]</fail>",failmsg);
      printf("request failed: %s\n",failmsg);
    }
  }
  close(sock);
}

