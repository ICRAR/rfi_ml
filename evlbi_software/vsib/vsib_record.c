/* wr.c -- Read/write VLBI VSIB input/output board and disks. */

/*
 * $Log: vsib_record.c,v $
 * Revision 1.62  2012-08-01 02:25:31  phi196
 * Added 10bit support
 *
 * Revision 1.61  2012-03-09 04:31:59  phi196
 * Work with large files and remote recording
 *
 * Revision 1.60  2012-03-04 22:53:56  phi196
 * Simulate 1PPS missing
 *
 * Revision 1.59  2012-02-28 03:00:43  phi196
 * Large file tweaks
 *
 * Revision 1.58  2012-01-20 00:36:26  phi196
 * Added large file support
 *
 * Revision 1.57  2011-10-28 03:29:49  phi196
 * Add initial 32bit/16 channel support for sampler stats
 *
 * Revision 1.56  2011/06/07 23:3/3:49  phi196
 * Invert option
 *
 * Revision 1.55  2010/04/01 03:13:45  phi196
 * Minor memory leak on exit
 *
 * Revision 1.54  2009/10/20 22:50:45  phi196
 * Minor bug with 64 MHz sampler stats
 *
 * Revision 1.53  2009/05/28 04:30:52  phi196
 * Merge with curtin branch
 *

 * Revision 1.52  2009/03/26 04:02:00  phi196
 * Updated ipd handling
 *
 * Revision 1.51.2.4  2009/02/24 04:40:36  phi196
 * Fix at-vlba conversion
 *
 * Revision 1.51.2.3  2009/02/19 09:39:47  phi196
 * Convert AT to VLBA encoding for 64 MHz...
 *
 * Revision 1.51.2.2  2009/02/17 22:46:34  phi196
 * Trival bugs with 1bit conversion
 *
 * Revision 1.51.2.1  2009/02/17 04:56:59  phi196
 * Initial shuffle/1bit version
 *
 * Revision 1.51  2009/01/14 10:54:58  phi196
 * Actually add ipd option
 *
 * Revision 1.50  2008/12/18 12:47:53  phi196
 * Fix CRC problem(?)
 *
 * Revision 1.49  2008/12/15 10:53:56  phi196
 * Added ipd option and startup sleep
 *
 * Revision 1.48  2008/10/24 00:24:49  phi196
 * Change default port
 *
 * Revision 1.47  2008/07/04 06:06:45  phi196
 * Fix potential error in UDP mtu check
 *
 * Revision 1.46  2008/05/19 02:26:45  phi196
 * Was not updating MJD for mark5b modes
 *
 * Revision 1.45  2008/05/13 04:48:02  phi196
 * fixed xxooooxx and ooxxxxoo modes
 *
 * Revision 1.44  2008/05/09 00:11:54  phi196
 * Generalise split data a little for 256 Mbps modes
 *
 * Revision 1.43  2008/04/18 06:23:30  phi196
 * Fix UDP bug. Allow TCP windowto not be set
 *
 * Revision 1.42  2008/03/28 08:17:56  phi196
 * BUG with PPS skip fixed for net2 case. Added port2 option
 *
 * Revision 1.41  2008/03/25 00:36:09  phi196
 * Merged from UDP branch
 *
 * Revision 1.40  2007/09/10 20:43:12  phi196
 * Allow non-integer bandwidth
 *
 * Revision 1.39.4.9.2.8  2008/03/19 11:42:23  phi196
 * Added xx0000xx and 00xxxx00 compress modes
 *
 * Revision 1.39.4.9.2.7  2008/01/10 02:26:31  phi196
 * Functionalised stats mark5b headers. Potential fix for bigbuffer overflow
 *
 * Revision 1.39.4.9.2.6  2007/11/11 06:02:42  phi196
 * Added split mode evlbi
 *
 * Revision 1.39.4.9.2.5  2007/09/07 12:58:04  phi196
 * Fixed sequence #
 *
 * Revision 1.39.4.9.2.4  2007/09/07 08:11:51  phi196
 * Assummed passed udp size is MTU not data size
 *
 * Revision 1.39.4.9.2.3  2007/09/07 07:04:31  phi196
 * dd
 *
 * Revision 1.39.4.9.2.2  2007/09/06 09:48:38  phi196
 * Modified approach to udp send
 *
 * Revision 1.39.4.9.2.1  2007/09/06 07:40:50  phi196
 * Initial UDP changes
 *
 * Revision 1.39.4.9  2007/08/07 12:06:26  phi196
 * Did not account for compression
 *
 * Revision 1.39.4.8  2007/07/27 04:54:16  phi196
 * Don't force mark5b
 *
 * Revision 1.39.4.7  2007/07/26 13:37:08  phi196
 * mjd fix
 *
 * Revision 1.39.4.6  2007/07/26 13:21:53  phi196
 * mjd debug
 *
 * Revision 1.39.4.5  2007/07/26 11:35:48  phi196
 * Major frame count error
 *
 * Revision 1.39.4.4  2007/06/23 09:22:22  phi196
 * Fixed dumy data send
 *
 * Revision 1.39.4.3  2007/06/22 06:27:06  phi196
 * Cope with 1PPS drops and drop data when buffer too full
 *
 * Revision 1.39.4.2  2007/06/21 11:55:33  phi196
 * updates
 *
 * Revision 1.39.4.1  2007/06/13 01:55:11  phi196
 * Initial mark5b mods
 *
 * Revision 1.39.6.2  2007/06/22 05:41:58  phi196
 * Logic buf
 *
 * Revision 1.39.6.1  2007/06/20 01:23:03  phi196
 * Skip files when behind
 *
 * Revision 1.39  2007/05/03 01:12:42  phi196
 * Cleaned up 1sec offset
 *
 * Revision 1.38  2007/05/01 06:10:11  phi196
 * 1 sec offset debugging
 *
 * Revision 1.37  2007/02/27 23:36:03  phi196
 * Fixed options
 *
 * Revision 1.36  2007/01/02 04:58:06  phi196
 * Added long options
 *
 * Revision 1.35  2007/01/02 04:19:35  phi196
 * Print the accumulated stats
 *
 * Revision 1.34  2006/12/21 00:04:30  phi196
 * Couple of minor bugs
 *
 * Revision 1.33  2006/12/13 04:03:51  phi196
 * Add check of bigbuffer usage and time offset to force restart and initial stats calculation
 *
 * Revision 1.32  2006/12/07 04:20:17  phi196
 * Dead reckon time from initial start. Use time_t as start epoch rather than
 * struct tm. Force gmt as time standard rather than odd mix of local and UT
 *
 * Revision 1.31  2006/12/05 06:00:45  phi196
 * Bug in figuring out how long to wait for round starttime
 *
 * Revision 1.30  2006/12/05 05:43:05  phi196
 * Add option to force start in integration boundary. Clean up usage of 
 * waituntil call to avoid possible race conditions
 *
 * Revision 1.29  2006/12/05 03:35:50  phi196
 * Added more compress modes
 *
 * Revision 1.28  2006/12/05 03:19:18  phi196
 * Merged from net branch. Added evlbi option
 *
 * Revision 1.27  2006/08/21 01:56:29  phi196
 * Update version number
 *
 * Revision 1.26  2006/08/21 01:49:51  phi196
 * Add channel 2+4 compression
 *
 * Revision 1.25  2006/08/21 01:44:23  phi196
 * Add sequence number
 *
 * Revision 1.24  2006/08/01 11:25:58  phi196
 * Minor update for vheader change
 *
 * Revision 1.23  2006/07/11 04:24:39  phi196
 * Added string access to time header
 *
 * Revision 1.22  2006/05/30 07:23:34  phi196
 * Added xo and ox compress mode
 *
 * Revision 1.21  2006/03/09 02:48:14  phi196
 * Fixed 64 MHz mode
 *
 * Revision 1.20  2006/03/08 13:31:44  phi196
 * Change number of channels based on mode
 *
 * Revision 1.19  2006/03/06 12:35:38  phi196
 * Problem setting time header
 *
 * Revision 1.18  2006/02/10 10:16:41  phi196
 * New header support
 *
 * Revision 1.17.2.10  2006/07/11 04:28:16  phi196
 * Wait until next scan boundary if late starting timed start
 *
 * Revision 1.17.2.9  2006/05/30 09:55:21  phi196
 * Forgot about header size
 *
 * Revision 1.17.2.8  2006/05/30 09:51:43  phi196
 * Adjust filesize for compression modes
 *
 * Revision 1.17.2.7  2006/05/30 09:20:48  phi196
 * Added filename and filesize to network stream
 *
 * Revision 1.17.2.6  2006/03/16 08:01:48  phi196
 * Allow window size to be set
 *
 * Revision 1.17.2.5  2006/03/14 10:54:36  phi196
 * Set header time in net mode
 *
 * Revision 1.17.2.4  2006/03/14 07:59:42  phi196
 * Added header support
 *
 * Revision 1.17.2.3  2006/03/12 13:05:38  phi196
 * Larger window
 *
 * Revision 1.17.2.2  2006/02/09 04:46:22  phi196
 * Read remote hostname from command line
 *
 * Revision 1.17.2.1  2006/02/08 22:38:09  phi196
 * Initial TCP networking
 *
 * Revision 1.17  2005/12/23 00:21:33  phi196
 * Minor bug with narrow bands. Check 1 PPS every second. Remove non-used 
 * header stuff. Save filenames to shared memory. Minor clean ups
 *
 * Revision 1.16  2005/10/27 04:21:52  phi196
 * Check all bits for PPS marker for 16 and 32 bit modes
 *
 * Revision 1.15  2005/09/29 01:35:49  phi196
 * Pointer bug
 *
 * Revision 1.14  2005/09/22 06:14:29  phi196
 * Time interval can be float and added (compile time) mode to turn off PSS 
 * check
 *
 * Revision 1.13  2005/09/20 00:11:20  phi196
 * Changed stderr to stdout and force stdout to line buffering (for disko)
 *
 * Revision 1.12  2005/09/12 05:39:16  phi196
 * Convert Zero's to Ohs and 1s to x
 *
 * Revision 1.11  2005/05/09 06:53:22  phi196
 * Clean up some debugging info
 *
 * Revision 1.10  2005/04/14 04:47:47  phi196
 * Added starttime option and autorestart
 *
 * Revision 1.9  2005/03/29 00:22:45  phi196
 * Added starttime option
 *
 * Revision 1.8  2005/03/24 05:30:13  phi196
 * Put compression code in own subroutine
 *
 * Revision 1.7  2005/03/22 04:33:25  phi196
 * Cleaned up code a little ready for auto-restart on 1PPS slip
 *
 * Revision 1.6  2005/03/15 05:42:32  phi196
 * Added hours to -t option
 *
 * Revision 1.5  2005/03/01 09:24:41  phi196
 * Added autofile name(time), default vsib device and bigbuf usage
 *
 * Revision 1.4  2005/02/23 23:29:46  wes128
 * Added <16Mhz support for recording - changed variable rate to clockrate
 *
 * Revision 1.3  2005/01/10 04:15:25  phi196
 * Fixed 1PPS missing bug
 *
 * Revision 1.2  2004/11/10 06:44:03  phi196
 * Forced to write mode
 *
 * Revision 1.1  2004/11/10 03:44:39  phi196
 * Changed name
 *
 * Revision 1.1  2004/11/10 02:53:12  phi196
 * Initial CVS version
 *
 * Revision 1.13  2003/03/05 08:01:56  amn
 * March 2003 Jb2/Mh Mk4/5-style Gbit test version.
 *
 * Revision 1.12  2002/11/29 10:58:40  amn
 * Unified ft2 and ft3; made mrg seq. numbered file opening work; wr parms.
 *
 * Revision 1.11  2002/11/07 09:06:49  amn
 * First Mk5A 1Gbit/s tests with Jodrell.
 *
 * Revision 1.10  2002/11/01 13:19:58  amn
 * Just before changing to use a common tstamp.cc file.
 *
 * Revision 1.9  2002/09/20 07:39:15  amn
 * Added mrg.c, merge/split for multi-PC files.
 *
 * Revision 1.8  2002/09/02 16:39:47  amn
 * Test version suitable for 50MHz/2 test pattern recording, for pb/JB.
 *
 * Revision 1.7  2002/08/09 11:26:56  amn
 * Jul-2002 first fringes Dwingeloo test version.
 *
 * Revision 1.5  2002/06/15 10:07:31  amn
 * Added shared memory and seek support.
 *
 * Revision 1.4  2002/06/14 13:02:34  amn
 * Dwingeloo test trip version.
 *
 */

/* Copyright (C) 2001--2002 Ari Mujunen, Ari.Mujunen@hut.fi

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software Foundation,
   Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.  */

#define _LARGEFILE_SOURCE 
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64

#ifdef DMALLOC
#include "dmalloc.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <errno.h>

#include <unistd.h>
#include <time.h>
#include <math.h> /* drem() */

#include <sys/types.h>  /* open() */
#include <sys/stat.h>  /* open() */
#include <fcntl.h>  /* open() */

#include <getopt.h>

#include <sys/time.h>  /* gettimeofday() */
#include <unistd.h>  /* gettimeofday(), usleep(), getopt() */

#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>

extern char *optarg;
extern int optind, opterr, optopt;

#include <string.h>  /* strstr(),strtok() */

#include <sys/ipc.h>  /* for shared memory */
#include <sys/shm.h>

#include <sys/ioctl.h>
#include "vsib_ioctl.h"

#include "tstamp.h"
#include "vheader.h"
#include "sh.h"

#define STRLEN 256

#define FALSE 0
#define TRUE 1

#define MINSTART_USEC 300000
#define MAXSTART_USEC 800000

#define PERCENT_TIMEOFFSET 70
#define PERCENT_MAXBIGBUF_USAGE 50

#define SECONDSABOVETHRESHOLD 30
#define EVLBIBEHINDTHRESHOLD 8

#define STATS_TIME 0.1  // Number of seconds to do stats calculation

#define UPDATE_TIME 10 // Update stats etc every 10 seconds (or once per file)

#define MK5B_FRAMESIZE 10000
#define MK5B_HEADERSIZE (4*sizeof(u_int32_t))

/* VSIB recording modes */

/* Depending on readMode either standard input or output. */
int vsib_fileno;

/* Simulate 1PPS failures */
int simulate_1pps = 0;

typedef enum {COMPNONE=0, COMP_000X=1, COMP_00X0=2, COMP_00XX=3, COMP_0X00=4, COMP_0X0X=5,
	      COMP_0XX0=6, COMP_0XXX=7, COMP_X000=8, COMP_X00X=9, COMP_X0X0=10, COMP_X0XX=11,
	      COMP_XX00=12, COMP_XX0X=13, COMP_XXX0=14, COMPALL=15, COMP_0X=-1, COMP_X0=-2,
	      COMP_00XXXX00=60, COMP_XX0000XX=195}
  compresstype;

void start_VSIB(int mode, int skip);
void start_recorder (int mode, int skip, time_t *starttime, float fileint,
		     int rountStart);
void stop_recorder (int *usleeps);
void waituntil (double starttime, float fileint);
double tim(void);
time_t now ();
void compress_data (char *mem, compresstype c_mode, size_t nread, size_t *nwrite);
void splitdata(int mode, char *mem1, char *mem2, size_t n);
int setup_net(char *hostname, int port, int window_size, int udp, int *sock);
int netsend(int sock, char *buf, size_t len);
int udpsend(int sock, int datagramsize, char *buf, int *bufsize, 
	    unsigned long long *sequence, int ipd);
void my_usleep(double usec);
double tm2mjd(struct tm date);
void init_bitreversal();
void convert_init(float bandwidth);
void invert_init(float bandwidth, int nchan);
void invert_data(char *p, size_t n);
void convertto1bit(char *p, size_t n);
void convertto16bit(char *p, size_t n);
unsigned short reversebits16 (unsigned short s);
unsigned int crcc (unsigned char *idata, int len, int mask, int cycl);
void accum_stats(char *p, size_t nread, unsigned long *stats, 
		 int bytespersample);
void print_stats (unsigned long stats[], int bytespersample, float bandwidth,
		  int active_stats[]);
void send_mark5bheader (int fid, int sock, u_int32_t mk5bheader[], double *mjd, 
			size_t nwrite, int net, int udp, int donetsend, 
			int *udpbufsize, char *p, int framepersec);
void shuffle_init(float bandwidth, int mode, int bitconvert);
void shuffle_data (char *p, size_t n, float bandwidth, int mode);
void kill_hup(int sig);

#define TIMESTART(i) \
        { dirs[i].prevOperTime = tim(); }
#define TIMESTOP(i) \
      { double dur = tim() - dirs[i].prevOperTime; \
        if (dur > dirs[i].maxOperTime) dirs[i].maxOperTime = dur; }

/* A protected (error-checking) ioctl() for VSIB driver. */
static void
vsib_ioctl(
  unsigned int mode,
  unsigned long arg
) {
  if (ioctl(vsib_fileno,
            mode,
            arg)
  ) {
    char *which;
    char err[255];
    
    if (vsib_fileno == STDOUT_FILENO) {
      which = "rd";
    } else {
      which = "wr";
    }
    snprintf(err, sizeof(err), "%s: ioctl(vsib_fileno, 0x%04x,...)", which, 
	     mode);
    perror(err);
    printf("%s: standard I/O is not an VSIB board\n", which);
    exit(EXIT_FAILURE);
  }
}  /* vsib_ioctl */


typedef struct sDir {
  char *nameTemplate;
  int nameNumber;
  int blocksPerFile;
  int fid;
  int nowBlocks;
  double prevOperTime;
  double maxOperTime;
} tDir, *pDir;

static unsigned char bytetable[256];
static unsigned short *shuffle_shorttable;
static unsigned char *shuffle_bytetable;
static unsigned char *convert1bit_table;
static uint32_t mask; // Mask for inverting spectrum

int main(int argc, char *argv[]) {
  // char *wr_version="1.0.1"; // Added 1PPS check
  // char *wr_version="1.0.2"; // Added Cmode to the header
  // char *wr_version="1.0.3"; // Added a bigbuffer check
  // char *wr_version="1.0.4"; // Increase the effect of "verbose" mode
  // char *wr_version="1.0.5"; // Cleaned up a little and added headers as 
                               // an option
  // char *wr_version="1.0.6"; // Bandwidth adjustable to less than 16 MHz - cwest
  // char *wr_version="1.0.7";   // Added starttime option and autorestart - cjp
  // char *wr_version="1.8";     // Various stuff
  //char *wr_version="1.9";      // Added extended headers
  //char *wr_version="1.10";      // Merged evlbi changes from net branch
  //char *wr_version="1.10";      // Merged evlbi changes from net branch
  //char *wr_version="1.11";      // Added more compress modes
  //char *wr_version="1.10";      // Add option for forcing start on scan boundary
  //char *wr_version="1.11";      // Dead reckon header time based on start time
  //char *wr_version="1.12";      // Check time offset and bigbuf not full for too long

  //char *wr_version="1.13";      // Fixed 1 sec offset in manual start mode
  //char *wr_version="1.14a";      // Mark5b option
  char *wr_version="1.15";      // USB

  int readMode;
  int blksize=32000;
  int totalblks=3600; /* 1 hour */
  char *memblk;
  char *p2 = NULL;
  char *p,*fileout=NULL;
  char *vsib_device=NULL;
  int ndirs=1;
  pDir dirs;
  int i, j;
  double recordStart;
  double totalDur;
  int usleeps;
  int bytespersample=0;
  int blocksPerSec;
  int vsib_started = 0;
  int blk=60; /* 60 seconds */
  int udpbufsize = 0;
  float fileint = 0.0; /* Remember file size in seconds */
  float datarate; /* Data rate before compression in bytes/sec */
  float bigbuf_sec;    /* Bigbuffer size on seconds, for this data rate */
  double mbits;
  int clockrate=32; // Default rate 32 MHz
  float bandwidth=16; // Default bandwidth 16MHz - for values less than 16Mhz 
                    // sample skipping is needed.
  int skip=0;       // Used for bandwidth < 16MHz
  int nchan=4;      // Number of channels recorded

  char *cptr;
  int tb_in_sec=TRUE,fb_in_sec=TRUE;
  int roundStart = FALSE;
  int mark5b = FALSE;
  int invert = FALSE;
  int nbits = 2;
  int c_mode=COMPALL;
  int mode=3;
  int udp = 0;
  int ipd = 0;
  int cfact = 1;
  int net=0;
  int net2=0;
  int port=52100;
  int port2=0;
  int shuffle=0;
  int startup_sleep=0;
  int window_size = -1;        /* 2 MB */
  int sock;
  int sock2;
  int status;
  int scale;
  int opt;
  int newfile=FALSE;
  int autorestart=TRUE;
  int verbose = TRUE;
  int check_stats = TRUE;
  int donetsend = 1;
  int restart;
  int ntimes_bigbuf_full=0;
  int ntimes_offset_large=0;
  int timeOK;
  int nbuf_stats=0;
  int active_stats[8];
  int update_time=UPDATE_TIME;
  unsigned long bigbuf_size;
  unsigned long *stats=0;
  unsigned long long udpsequence = 0;
  u_int32_t mk5bheader[4];
  unsigned short *nframe;
  int framepersec;
  double mjd;
  char hostname[STRLEN] = ""; /* Host name to send data to */
  char hostname2[STRLEN] = ""; /* Second Host name to send data to */
  char *headbuf;
  vhead *header;    // File header object

  /* Shared memory unique identifier (key) and the returned reference id. */
  key_t shKey;
  int shId = -1;
  int sequence = 0;
  ptSh sh;

  char datestr[100], timestr[100], msg[100];
  struct tm *date;
  time_t starttime=0, currenttime=0, firsttime;

  struct option options[] = {
    {"rate", 1, 0, 'r'},
    {"bandwidth", 1, 0, 'w'},
    {"window", 1, 0, 'W'},
    {"tcpwindow", 1, 0, 'W'},
    {"port", 1, 0, 'p'},
    {"sleep", 1, 0, 'S'},
    {"port2", 1, 0, 'P'},
    {"blocksize", 1, 0, 'b'},
    {"blockperfile", 1, 0, 'f'},
    {"duration", 1, 0, 't'},
    {"time", 1, 0, 't'},
    {"ipd", 1, 0, 'i'},
    {"bits", 1, 0, 'B'},
    {"compression", 1, 0, 'c'},
    {"outfile", 1, 0, 'o'},
    {"device", 1, 0, 'e'},
    {"mode", 1, 0, 'm'},
    {"host", 1, 0, 'H'},
    {"host2", 1, 0, 'J'},
    {"starttime", 1, 0, 's'},
    {"verbose", 0, 0, 'v'},
    {"round", 0, 0, 'x'},
    {"mk5b", 0, 0, 'M'},
    {"mark5b", 0, 0, 'M'},
    {"1bit", 0, 0, '1'},
    {"invert", 0, 0, 'I'},
    {"udp", 1, 0, 'u'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  setenv("TZ", "", 1); /* Force mktime to return gmt not local time */
  tzset();

  while ((opt = getopt_long_only(argc, argv, "s:e:w:r:t:b:f:W:p:m:c:H:d:i:o:vhx", 
				 options, NULL)) != EOF)
    switch (opt) {
    case 'w': // recording bandwidth
      bandwidth=atof(optarg);
      if(verbose) printf("Bandwidth: %.1f MHz\n", bandwidth);
      break;

    case 'W': // Window size
      window_size=atoi(optarg)*1024;
      if(verbose) printf("Windowsize: %d kBytes\n", window_size);
      break;

    case 'r': // recording rate
      clockrate=atoi(optarg);
      if (verbose) printf("Rate: %d\n",clockrate);
      if (!clockrate) exit(printf("Rate given (%s) is zero\n",optarg));
      break;

    case 'b': // block size
      blksize=atoi(optarg);
      if (verbose) printf("Blocksize: %d\n",blksize);
      if (!blksize) 
	exit(printf("Block size given (%s) is zero\n",optarg));
      break;

    case 'f': // blocks per file
      if (optarg[strlen(optarg)-1]=='s') 
	fb_in_sec=TRUE;
      else  
	fb_in_sec=FALSE;
      blk=atoi(optarg);
      if (verbose) printf("Blocks per file: %s\n",optarg);
      if (!blk) 
	exit(printf("Blocks per file given (%s) is zero\n",optarg));
      break;

    case 't': // seconds to record
      scale=1;
      if (optarg[strlen(optarg)-1]=='s') 
	tb_in_sec=TRUE;
      else if (optarg[strlen(optarg)-1]=='h') {
	scale=60*60; /* Hours -> seconds  */
	tb_in_sec=TRUE;
      } else if (optarg[strlen(optarg)-1]=='m') {
	scale=60; /* Minutes -> seconds  */
	tb_in_sec=TRUE;
      } else  
	tb_in_sec=FALSE;

      totalblks=ceil(atof(optarg)*scale);
      if (verbose) printf("Recording time: %d\n",totalblks);

      if (!totalblks) 
	exit(printf("Time to record (%s) is zero\n",optarg));
      break;

    case 'd': // Number of directories
      ndirs=atoi(optarg);
      if (verbose) printf("Number of directories: %d\n",ndirs);
      if (!ndirs) 
	exit(printf("Number of directories to record (%s) is zero\n",
		    optarg));
      break;

    case 'c': // Compress on the fly
      // First convert Zeros to Ohs and 1s to x
      for (cptr=optarg; *cptr!=(char)0; cptr++) {
	if (*cptr=='0')
	  *cptr = 'o';
	else if (*cptr=='O')
	  *cptr = 'o';
	else if (*cptr=='1')
	  *cptr = 'x';
      }

      if (!strcmp(optarg,"oooo")) {
	c_mode=COMPNONE; nchan=0; printf("No data!\n");

      } else if (!strcmp(optarg,"ooox")) { 
	c_mode=COMP_000X; cfact=4; nchan=1; 
	if (verbose) printf("Channel one only\n");

      } else if (!strcmp(optarg,"ooxo")) { 
	c_mode=COMP_00X0; cfact=4; nchan=1; 
	if (verbose) printf("Channel two only\n");

      } else if (!strcmp(optarg,"ooxx")) { 
	c_mode=COMP_00XX; cfact=2; nchan=2; 
	if (verbose) printf("Channel one & two\n");

      } else if (!strcmp(optarg,"oxoo")) { 
	c_mode=COMP_0X00; nchan=1; 
	if (verbose) printf("Channel three only\n");

      } else if (!strcmp(optarg,"oxox")) { 
	c_mode=COMP_0X0X; cfact=2; nchan=2; 
	if (verbose) printf("Channel one & three\n");

      } else if (!strcmp(optarg,"oxxo")) { 
	c_mode=COMP_0XX0; cfact=2; nchan=2; 
	if (verbose) printf("Channel two & three\n");
	exit(printf("Not yet supported!\n"));

      } else if (!strcmp(optarg,"oxxx")) { 
	c_mode=COMP_0XXX; cfact=0; nchan=3; 
	if (verbose) printf("Channel one, two & three\n");
	exit(printf("Not yet supported!\n"));

      } else if (!strcmp(optarg,"xooo")) { 
	c_mode=COMP_X000; cfact=4; nchan=1; 
	if (verbose) printf("Channel four only\n");

      } else if (!strcmp(optarg,"xoox")) { 
	c_mode=COMP_X00X; cfact=2; nchan=2; 
	if (verbose) printf("Channel one & four\n");
	exit(printf("Not yet supported!\n"));

      } else if (!strcmp(optarg,"xoxo")) { 
	c_mode=COMP_X0X0; cfact=2; nchan=2;  
	if (verbose) printf("Channel two & four\n");

      } else if (!strcmp(optarg,"xoxx")) { 
	c_mode=COMP_X0XX; cfact=-1; nchan=3; 
	if (verbose) printf("Channel one, two & four\n");
	exit(printf("Not yet supported!\n"));

      } else if (!strcmp(optarg,"xxoo")) { 
	c_mode=COMP_XX00; cfact=2; nchan=2; 
	if (verbose) printf("Channel three & four\n");

      } else if (!strcmp(optarg,"xxox")) { 
	c_mode=COMP_XX0X; cfact=-1; nchan=3; 
	if (verbose) printf("Channel one, three & four\n");
	exit(printf("Not yet supported!\n"));

      } else if (!strcmp(optarg,"xxxo")) { 
	c_mode=COMP_XXX0; cfact=-1; nchan=3; 
	if (verbose) printf("Channel two, three & four\n");
	exit(printf("Not yet supported!\n"));

      } else if (!strcmp(optarg,"ox")) { 
	c_mode=COMP_0X; cfact=2; nchan=2; 

	if (verbose) printf("Dropping every second byte\n");
      } else if (!strcmp(optarg,"xo")) { 
	c_mode=COMP_X0; cfact=2; nchan=2; 

	if (verbose) printf("Dropping every second byte\n");
      } else if (!strcmp(optarg,"xx")) { 
	c_mode=COMPALL; cfact=1; nchan=4; 
	if (verbose) printf("No compression\n");

      } else if (!strcmp(optarg,"xxxx")) { 
	c_mode=COMPALL; cfact=1; nchan=4; if (verbose) printf("No compression\n");

      } else if (!strcmp(optarg,"ooxxxxoo")) { 
	c_mode=COMP_00XXXX00; cfact=2; nchan=2; if (verbose) printf("Channel 3,4,5,6\n");

      } else if (!strcmp(optarg,"xxooooxx")) { 
	c_mode=COMP_XX0000XX; cfact=2; nchan=2; if (verbose) printf("Channel 1,2,7,8\n");

      } else { 
	exit(printf("Unknown channel mode %s\n", optarg)); 
      }
      break; 

    case 'o': // file to record
      fileout=(char *) strdup(optarg);
      if (verbose) printf("Disk File: %s\n", fileout);
      if (!fileout) 
	exit(printf("File to record to (%s) is zero\n", optarg));
      break;

    case 'H': // Remote host to connect to
      if (strlen(optarg)>STRLEN-1) {
	printf("Warning: Remote host (%s) too long\n", optarg);
	exit(1);
      } 
      strcpy(hostname, optarg);
      net = 1;
      if (verbose) printf("Remote host: %s\n", hostname);
      break;

    case 'J': // Second remote host to connect to
      if (strlen(optarg)>STRLEN-1) {
	printf("Warning: Remote host (%s) too long\n", optarg);
	exit(1);
      } 
      strcpy(hostname2, optarg);
      net2 = 1;
      if (verbose) printf("Second remote host: %s\n", hostname2);
      break;

    case 'e': // vsib device file
      vsib_device=(char *) strdup(optarg);
      if (verbose) printf("Vsib device: %s\n", vsib_device);
      if (!vsib_device) 
	exit(printf("Error setting vsib device to \"%s\"\n", optarg));
      break;

    case 'm': // mode to record
      mode=atoi(optarg);
      if (verbose) printf("Mode: %d\n", mode);
      if (mode<0 || mode>13) 
	exit(printf("invalid mode %s\n", optarg));
      if (mode>4) 
	printf("invalid mode %s\n", optarg);
      break;

    case 's': // start time
      /* Get and convert target UTC time to Unix seconds. */
      if (getDateTime(optarg, &starttime)) {
	printf("Failed to convert ISO 8601 UTC date/time `%s'\n", 
	       optarg);
	exit(EXIT_FAILURE);
      }
      break;

    case 'x': // Force start on "round" time
      roundStart=TRUE;
      if (verbose) printf("Force start on integration boundary\n");
      break;

    case 'M':
      mark5b=TRUE;
      if (verbose) printf("Convert to Mark5b format\n");
      break;

    case 'B':
      nbits=atoi(optarg);
      if (verbose) printf("Bits: %d\n", nbits);
      break;

    case 'I':
      invert=1;
      if (verbose) printf("Inverting data\n");
      break;

    case 'u': // UDP datagramsize
      udp=atoi(optarg);
      if (verbose) printf("UDP: %d\n", udp);
      break;

    case 'i': // Interpacket Delay
      ipd=atoi(optarg);
      if (verbose) printf("IPD: %d\n", ipd);
      break;

    case 'p':
      port = atoi(optarg);
      if (verbose) printf("Port: %d\n", port);
      net = 1;
      break;

    case 'P':
      port2 = atoi(optarg);
      if (verbose) printf("Port2: %d\n", port2);
      break;

    case 'S':
      startup_sleep = atoi(optarg);
      if (verbose) printf("Sleep: %d\n", startup_sleep);
      break;

    case 'v': // verbose
      verbose=TRUE;
      if (verbose) printf("Verbose mode\n");
      break;

    case 'h': // help or unknown
    case '?':
      exit(printf(
		  "usage : %s -r rate(MHz) -t recording_time(blks/secs/hours) "
		  "-b blksize -f blk -o filenameprefix -c compact_mode "
		  "-m mode -h (help) -v (verbose) "
		  "-w bandwidth(MHz)\nVersion %s\n", argv[0],
		  wr_version));
    }

  switch (mode) {
  case 0:
    bytespersample = 4;
    break;
  case 2:
    bytespersample = 2;
    break;
  case 3:
    bytespersample = 1;
    break;
  default:
    printf("Error: Unsupported recording mode %d\n", mode);
    bytespersample = 1;
    break;
  }

  if (nbits==1) {
    cfact*=2;
    convert_init(bandwidth);
  } else if (nbits !=2 && nbits !=8 && nbits != 10) {
    printf("Unsupported number of bits %d\n", nbits);
    exit(1);
  }

  if (port2==0) port2 = port;

  if (bandwidth*2 < clockrate){
    skip = (clockrate/bandwidth/2) - 1;
    printf("Skipping %d samples for bandwidth %.1f MHz\n", skip, bandwidth);
  }
  datarate = clockrate*1e6*bytespersample/(skip+1);

  if (!mark5b) udp = 0;

  if (mark5b && bandwidth==64) shuffle=1;

  if (shuffle) shuffle_init(64, mode, 1);

  if (invert) invert_init(bandwidth, nchan);

  // Use pointer into header for frame number and crc
  nframe = (unsigned short*)&mk5bheader[1];
  framepersec = datarate / MK5B_FRAMESIZE/cfact;
  
  if (mark5b) {
    printf("Warning: Forcing 10k block size for Mark5b mode\n");
    blksize = MK5B_FRAMESIZE*cfact;    
    
    // Mark5b Sync word is constant. Blank user bits
    mk5bheader[0] = 0xABADDEED;
    mk5bheader[1] = 0xF00F0000;
    mk5bheader[2] = 0x0;
    mk5bheader[3] = 0x0;
    
    *nframe = 0;
    
    init_bitreversal();
    
    if (udp) {
      udp -= 20; // IP header
      udp -= 4*2; // UDP header
      udp -= sizeof(long long); // Sequence number
      udp &= ~0x7;  //Truncate to smallest multiple of 8

      printf("UDP data size %d\n", udp);
      
      if (udp<=0) {
	printf("Error: Specified UDP MTU size (%d) too small\n", udp+20+4*2
	       +(int)sizeof(long long));
	exit(1);
      }
    }
  }

  if (tb_in_sec) totalblks=totalblks*datarate/blksize;
  
  if (fb_in_sec) {
    fileint = blk;
    blk = blk*datarate/blksize;
  } else
    fileint = blk*blksize/datarate;
  if (fileint<update_time) 
    update_time = fileint;
  
  blocksPerSec = datarate/blksize;
  nbuf_stats = ceil(blocksPerSec*STATS_TIME); 
  
  //  /* Force start time on sensible integration boundary */
  if (roundStart && starttime!=0) {
    printf("Warning: Ignoring round start as explicit start time given\n");
  //    double nint;
  //    time_t  dUT0;
  //    
  //    dUT0 = starttime % 86400;
  //    nint = floor((float)dUT0/fileint);
  //    
  //    starttime = (starttime-dUT0) + nint*fileint;
  }
  
  if (verbose) printf("Total number of blocks: %d\n", totalblks);
  if (verbose) printf("blocksize: %d\n", blksize);
  if (verbose) printf("No. blocks per file: %d\n", blk);
  
  if (!fileout) fileout=strdup("test");
  if (!vsib_device) vsib_device=strdup("/dev/vsib");
  
  /* Which channels are active after compression? */
  for (i=0; i<8; i++)
    active_stats[i] = 0;

  if (bandwidth<=16) {
    if (c_mode>COMPNONE) {
      for (i=0; i<4; i++)
 	if (c_mode & 1<<i) {
 	  active_stats[i] = 1;
 	  active_stats[i+4] = 1;
 	}
    } else {
      if (c_mode==COMP_0X) 
 	for (i=0; i<4; i++)
 	  active_stats[i] = 1;
      else if (c_mode==COMP_X0) 
 	for (i=4; i<8; i++)
 	  active_stats[i] = 1;
    } 
    
  } else if (bandwidth==32) {
    if (c_mode==COMP_0X) {
      active_stats[0] = 1;
      active_stats[1] = 1;
    } else if (c_mode==COMP_X0) {
      active_stats[2] = 1;
      active_stats[3] = 1;
    } else if (c_mode==COMP_00XX) {
      active_stats[0] = 1;
      active_stats[2] = 1;
    } else if (c_mode==COMP_XX00) {
      active_stats[1] = 1;
      active_stats[3] = 1;
    } else if (c_mode==COMPALL) {
      for (i=0;i<4; i++) 
	active_stats[i] = 1;
    } else if (c_mode != COMPNONE) 
      printf("Warning: Compression mode will corrupt data!\n");
  } else if (bandwidth==64) {
    if (c_mode>COMPNONE && c_mode<COMPALL) {
      printf("Warning: Compression mode will corrupt data!\n");
    } else if (c_mode==COMP_0X) {
      active_stats[0] = 1;
    } else if (c_mode==COMP_X0) {
      active_stats[1] = 1;
    } else if (c_mode==COMPALL) {
      active_stats[0] = 1;
      active_stats[1] = 1;
    }
  }

  if (starttime!=0) {
    date = gmtime(&starttime); // will start on the next second
    strftime(datestr, 99, "%d/%m/%Y %H:%M:%S", date);

    printf("Will start recording at %s UT\n", datestr);
  }

  /* Init variables according to command line, allocate buffers. */
  /*readMode = (strstr(argv[0], "rd") != NULL);*/
  readMode = 0;

  if (readMode) {
    vsib_fileno = open(vsib_device, O_WRONLY|O_LARGEFILE);
  } else {
    vsib_fileno = open(vsib_device, O_RDONLY);
  }
  if (vsib_fileno == -1) {
    snprintf(msg, 100, "Trying to open %s", vsib_device);
    perror(msg);
    exit(0);
  }

  /* Find out total size of bigbuf area */
  vsib_ioctl(VSIB_GET_BIGBUF_SIZE, (unsigned long)&bigbuf_size);

  /* bigbufsize in seconds */
  bigbuf_sec = bigbuf_size/datarate;

  if (verbose) printf("Bigbuffer is %.1f seconds\n", bigbuf_sec);

  if (udp) {
    assert( (memblk = malloc(blksize+udp+MK5B_HEADERSIZE)) != NULL );
  } else {
    assert( (memblk = malloc(blksize)) != NULL );
  }

  if (net2) {
    assert( (p2 = malloc(blksize/2)) != NULL );
  }

  /* Allocate one extra struct for call-to-call time statistics. */
  assert( (dirs = malloc((ndirs+1)*sizeof(tDir))) != NULL );
  for (i=0; i<ndirs; i++) {
    if (!i) dirs[i].nameTemplate = strtok(fileout," ");
    else dirs[i].nameTemplate = strtok(NULL," ");
    if (!dirs[i].nameTemplate) 
      exit(printf("name template %d failed\n",i));
    dirs[i].nameNumber = 0;
    dirs[i].blocksPerFile = blk;
    dirs[i].nowBlocks = dirs[i].blocksPerFile;  /* init to end */
    dirs[i].maxOperTime = 0.0;  /* init to smallest value */
  }  /* for each directory */
  dirs[ndirs].nameTemplate =   "call-to-call  ";
  dirs[ndirs].maxOperTime = 0.0;

  if (nbits>2) check_stats = 0;

  if (check_stats) {
    /* Allocate memory for sampler statistics */
    stats = malloc(256*sizeof(long)*bytespersample);
    if (stats==NULL) {
      sprintf(msg, "Trying to allocate %d bytes", 
	      256*(int)sizeof(long)*bytespersample);
      perror(msg);
      return(1);
    }
  }

  /* Create and initialize shared memory. */
  shKey = fourCharLong('v','s','i','b');
  assert( (shId = shmget(shKey, sizeof(tSh), IPC_CREAT | 0777)) != -1 );
  assert( (sh = (ptSh)shmat(shId, NULL, 0)) != (void *)-1 );

  /* Initilize shared memory values */
  strcpy(sh->currentfile, ""); 

  if (mode==2) 
    nchan *= 2;
  else if (mode==0)
    nchan *= 4;

  if (bandwidth==64) 
    nchan /= 4;
  else if (bandwidth==32) 
    nchan /= 2;

  if (nbits==10)  // Really 16bits
    nchan /=8;
  

  /* Initialize the header object and fill in the static values */

  header = newheader();

  setantennaid(header, "Tt");
  setantennaname(header, "Test Antenna");
  setexperimentid(header, "VTXXXX");
  setrecorderversion(header, 1, 8);
  setbandwidth(header, bandwidth);
  if (net2) {
    setnchan(header, nchan/2);
  } else {
    setnchan(header, nchan);
  }
  if (nbits==10)
    setnumbits(header, 16);
  else
    setnumbits(header, nbits);
  setencoding(header, VLBA);

#if 0
  Opening the device special file already does this.
    For write() this open()-related (extra) DMA reset clears
    memory buffers and r/w pointers with arg==0.
    /* Init DMA. */
    vsib_ioctl(VSIB_RESET_DMA, 0);
#endif

  setlinebuf(stdout);
  
  /* Install a signal catcher to simulate 1PPS events*/
  signal(SIGHUP, kill_hup);

  if (net) {
    /* Connect to the receiving server */

    if (strlen(hostname) == 0) {
      strcpy(hostname, "localhost");
    }

    if (startup_sleep>0) sleep(startup_sleep);

    status = setup_net(hostname, port, window_size, udp, &sock);
    if (status) exit(1);

    if (net2) {
      status = setup_net(hostname2, port2, window_size, udp, &sock2);
      if (status) exit(1);
    }
     
  } else {
    if (net2) {
      printf("Ignoring -host2 option. Host not set\n");
      net2 = 0;
    }
  }

  /* Start the board.  (After opening it is still stopped.) */
  /* For disk write, start VSIB first and write to disk later. */
  /* For disk read, read first and start VSIB later. */

  /* If requested wait until the specified time */

  if (readMode) {
    vsib_started = 0;
  } else {
    start_recorder(mode, skip, &starttime, fileint, roundStart);
    firsttime=starttime;
  }

  /* Start looping for total number of blocks. */
  usleeps = 0;
  recordStart = tim();
  TIMESTART(ndirs);  /* call-to-call (not per file) */

  date = gmtime(&starttime); 
  mjd = tm2mjd(*date);
  //  nsec = 0;

  blk=0;
  while (blk<totalblks) {
    i = blk % ndirs;

    restart = FALSE;
    timeOK = TRUE;
    if (dirs[i].nowBlocks >= dirs[i].blocksPerFile) {
      /* Open a new file in this directory / template. */
      char pn[255];
      int n = blk / dirs[i].blocksPerFile;
      time_t time_offset;
      
      dirs[i].nowBlocks = 0;
      
      // Check that the difference between current time and the header
      // time is not greater than the size of the bigbuffer. If so we need
      // to restart the recorder and don't bother opening this file
      // Current time and starttime have been set at the end of this loop
      // when the previous file was closed
      
      donetsend = 1; // Reset to true at the start of each file
      time_offset = currenttime-starttime;
      if (time_offset>0) {
	
	printf("Warning: Header time is offset from wall clock time by %d "
	       "seconds\n", (int)(time_offset));
	
	if (time_offset>=bigbuf_sec) {
	  printf("ERROR: This is more than size of bigbuffer.\n");
	  printf("Restarting recorder\n");
	  restart = TRUE;
	  donetsend = 0;
	} else if (net && time_offset>EVLBIBEHINDTHRESHOLD) {
	  printf("ERROR: This is greater than EVLBIBEHINDTHRESHHOLD (%d)\n",
		 EVLBIBEHINDTHRESHOLD);
	  printf("Skipping next data file\n");
	  donetsend = 0;
	} else if (time_offset/bigbuf_sec*100>PERCENT_TIMEOFFSET) {
	  ntimes_offset_large++;
	  
	  // We are only checking once per file
	  if (ntimes_offset_large*fileint>SECONDSABOVETHRESHOLD) {
	    date = gmtime(&currenttime);
	    strftime(datestr, 99, "%Y%m%d:%H%M%S\n", date);
	    
	    donetsend = 0;
	    restart = TRUE;
	    printf("ERROR: Offset has been more than %d%% "
		   "of bigbuf size for %d seconds at %s\n", 
		   PERCENT_TIMEOFFSET, SECONDSABOVETHRESHOLD,
		   datestr);
	    
	    printf(" Restarting recorder\n");
	    
	  }
	} else 
	  ntimes_offset_large = 0;
      }
      
      date = gmtime(&starttime); // will start on the next second
      mjd = tm2mjd(*date);
      strftime(timestr, 99, "%j_%H%M%S", date);
      
      snprintf(pn, sizeof(pn), "%s_%s.lba", dirs[i].nameTemplate, timestr);
      //snprintf(pn, sizeof(pn), "%s", "/dev/null");
      
      if (n==0) TIMESTART(i);  /* only for first file */
      
      if (restart || (net && !donetsend)) {
	if (!restart) printf("Not sending header\n");
      } else {
	
	newfile=TRUE;
	settime(header, date);
	
	date = gmtime(&currenttime); // will start on the next second
	setfiletime(header, date);
	
	sequence++;
	header->sequence = sequence;
	
	if (net) {
	  if (!mark5b) {
	    unsigned long long filesize;
	    short fnamesize;
	    
	    writeheader(header, 0, &headbuf);
	    
	    /* Send network header */
	    filesize = (long long)dirs[i].blocksPerFile * (long long)blksize / cfact;
	    if (net2) filesize /= 2;
	    filesize += header->headersize;
	    fnamesize = strlen(pn)+1;
	    
	    status = netsend(sock, (char*)&filesize, sizeof(long long));
	    if (status) exit(1);
	    status = netsend(sock, (char*)&fnamesize, sizeof(short));
	    if (status) exit(1);
	    status = netsend(sock, pn, fnamesize);
	    if (status) exit(1);
	    status = netsend(sock, headbuf, header->headersize);
	    if (status) exit(1);
	    
	    if (net2) {
	      status = netsend(sock2, (char*)&filesize, sizeof(long long));
	      if (status) exit(1);
	      status = netsend(sock2, (char*)&fnamesize, sizeof(short));
	      if (status) exit(1);
	      status = netsend(sock2, pn, fnamesize);
	      if (status) exit(1);
	      status = netsend(sock2, headbuf, header->headersize);
	      if (status) exit(1);
	    }
	    
	    free(headbuf);
	  }
	} else { // Writing to disk
	  if (readMode) {
	    assert((dirs[i].fid=open(pn, O_RDONLY)) != -1 );
	  } else {
	    assert((dirs[i].fid=creat(pn, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH))!=-1);
	    
	    // Write the header into the data file
	    
	    if (!mark5b) writeheader(header, dirs[i].fid, NULL);
	  }
	} 
	system("date");
	
	printf("at block = %d, opened file '%s'\n", blk, pn);
	
	strcpy(sh->lastfile, sh->currentfile); // Remember last filename
	if (strlen(pn)>=SH_MAXSTR) {
	  printf("%s is too long to save to shared memory, increase MAXSTR in sh.h\n", pn);
	  strcpy(sh->currentfile, ""); 
	} else 
	  strcpy(sh->currentfile, pn); // Save filename to shared memory. 
	
      }
    }
    
    /* Write (or read) one block. */
    /* xxx: need to add '-1' error checks to VSIB read()/write() calls */
    if (udp) {
      p = memblk+udpbufsize+MK5B_HEADERSIZE;
    } else {
      p = memblk;
    }
    if (readMode) {
      size_t nwritten;
      
       assert( (read(dirs[i].fid,
		     p,
		     blksize)) == blksize );

       /* Write a block to VSIB; if not enough space, sleep a little. */
       nwritten = write (vsib_fileno, p, blksize);
       while (nwritten < (size_t) blksize) {
	 /* Not space for one full block in buffer, start VSIB transmission */
	 /* and wait for vacant space. */
	 if (readMode) {
	   if (!vsib_started) {
	     printf("starting vsib because bigbuf full\n");
	     /* xxx: reload PLX FIFOs */
	     vsib_ioctl(VSIB_RESET_DMA, /*skip_clear=>*/1);
	     start_VSIB(mode, skip);
	     vsib_started = 1;
	   }
	 }
	 usleep(1000);  /* a small amount, probably ends up to be 10--20msec */
	 usleeps++;
	 nwritten += write (vsib_fileno, p+nwritten, blksize-nwritten);
       }  /* while not at least one full block in VSIB DMA ring buffer */
       
    } else {
      size_t nread=0, nwrite;

      if (!restart) {
	/* Read a block from VSIB; if not enough, sleep a little. */
	nread = read (vsib_fileno, p, blksize);
	while (nread < (size_t) blksize) {
	  /* Not one full block in buffer, wait. */
	  usleep(1000);  /* a small amount, probably ends up to be 10--20msec */
	  usleeps++;
	  nread += read (vsib_fileno, p+nread, blksize-nread);
	}  /* while not at least one full block in VSIB DMA ring buffer */
	if (newfile) {
	  newfile=FALSE;
	  
	}
	
	// Check sampler statistic for first few blocks
	if (check_stats && (dirs[i].nowBlocks % (update_time*blocksPerSec) < nbuf_stats)) {
	  /* Check sampler statistics. If bytespersample==1 just accumulate all
	     possible bit patterns. Otherwise we need to accumulate each
	     byte seperately.  This will be interpreted later
	  */
	  
	  if (dirs[i].nowBlocks % (update_time*blocksPerSec)==0)  // Start of file, reset accumulator
	    for (j=0; j<256*bytespersample; j++)
	      stats[j] = 0;
	  //memset(stats, 0, 256*sizeof(long)*bytespersample);
	  
	  accum_stats(p, nread, stats, bytespersample);
	}
	
	if (check_stats && (dirs[i].nowBlocks % (update_time*blocksPerSec)==nbuf_stats-1)) { // Print stats
	  print_stats(stats, bytespersample, bandwidth, active_stats);
	}
	
	if (!(dirs[i].nowBlocks % blocksPerSec)) {  // 1 sec boundary
	  int ppsOK;
	  double thistime;
	  time_t tt;
	  
	  thistime = tim();
	  tt = floor(thistime+0.5);
	  
	  //printf("1 sec boundary at %.2f\n", thistime);
	  //printf("DEBUG: nowBlocks = %d\n", dirs[i].nowBlocks);
	  //printf("  %s\n", ctime(&tt));
	  
	  // Check PPS marker is present. Need to handle 1, 2 and 4 byte data
	  ppsOK = 0;
	  if ((*p&0xFF)==0xFF) {
	    if (mode==2 || mode==0) {
	      char *memptr;
	      memptr = p;
	      memptr++;

	      if ((*memptr&0xFF)==0xFF) {
		if (mode==0) {
		  memptr++;
		  if ((*memptr&0xFF)==0xFF) {
		    memptr++;
		    if ((*memptr&0xFF)==0xFF) {
		      ppsOK = 1;
		    }
		  }
		} else {
		  ppsOK = 1;
		}
	      }
	    } else {
	      ppsOK = 1;
	    }
	  }
	  if (simulate_1pps) {
	    simulate_1pps = 0;
	    ppsOK = 0;
	  }

	  if (ppsOK) {
	    if (verbose && dirs[i].nowBlocks % (update_time*blocksPerSec) ==0) {
	      if (dirs[i].nowBlocks!=0) system("date");
	      printf("1PPS OKAY\n");
	    }
	  } else if (autorestart) {
	    currenttime = now();
	    date = gmtime(&currenttime);
	    strftime(datestr, 99, "%Y%m%d:%H%M%S\n", date);

	    printf("\n\nERROR: 1PPS transition ABSENT (%02x) at %s\n",
		   *p,datestr);
	    printf(" Restarting recorder\n");
	    restart = 1;
	  }

	  if (ppsOK) { // Only check bigbuf if not restarting anyhow
	    float percentage_used;
	    unsigned long bigbuf_usage;

	    vsib_ioctl(VSIB_GET_BYTES_IN_BIGBUF, (unsigned long)&bigbuf_usage);
	    percentage_used = bigbuf_usage/(double)bigbuf_size*100;

	    if (verbose && dirs[i].nowBlocks % (update_time*blocksPerSec) ==0) { // New file
	      // Report bigbuffer usage
	      printf("MBytes left in the BIGBUF %lu (%2.0f%%)\n", 
		     bigbuf_usage/1024/1024, percentage_used); 
	    }

	    if (percentage_used<PERCENT_MAXBIGBUF_USAGE) {
	      ntimes_bigbuf_full++;

	      // As we check this every second, ntimes_bigbuf_full is the
	      // number of seconds the bigbuffer has been fuller than the
	      // threshold
	      if (ntimes_bigbuf_full>SECONDSABOVETHRESHOLD) {
		currenttime = now();
		date = gmtime(&currenttime);
		strftime(datestr, 99, "%Y%m%d:%H%M%S\n", date);

		restart = 1;
		printf("\n\nERROR: BIGBUF has been more than %d%% "
		       "full for %d seconds at %s\n", 
		       PERCENT_MAXBIGBUF_USAGE, SECONDSABOVETHRESHOLD,
		       datestr);
		printf(" Restarting recorder\n");

	      }

	    } else {
	      ntimes_bigbuf_full = 0;
	    }
	  }
	}
      }

      if (restart) {
	double thistime;
	int currentblocks;

	currentblocks = dirs[i].nowBlocks; // Remember 

	// First we must stop the recorder
	stop_recorder(&usleeps);

	// Need to re-initialise file struct to ensure files are re-opened
	for (j=0; j<ndirs; j++) {
	  dirs[j].nowBlocks = dirs[j].blocksPerFile;  /* init to end */
	}

	// Close and re-open device file
	if (close(vsib_fileno)) {
	  perror("Trying to close vsib device file");
	  exit(0);
	}

	if (net) { // Need to close and open socket
#if 0
	  status = close(sock);
	  if (status!=0) {
	    perror("Closing network connection");
	    exit(1);
	  }
	  status = setup_net(hostname, port, window_size, udp, &sock);
	  if (status) exit(1);
#endif
	  if (!mark5b && donetsend) {
	    long long sent = 0;
	    // We need to send bytes till the next "file" boundary

	    //printf("DEBUG: nowBlocks=%d\n", dirs[i].nowBlocks);
	    //printf("DEBUG: blocksPerFile=%d\n", dirs[i].blocksPerFile);

	    printf("Sending random data till next header boundary\n");
	    while (currentblocks < dirs[i].blocksPerFile) {
	      status = netsend(sock, p, blksize/cfact); 
	      sent += blksize/cfact;
	      currentblocks++;

	      if (net2) {
		status = netsend(sock2, p, blksize/cfact); 
		sent += blksize/cfact;
		currentblocks++;
	      }
	    }
	    printf("DEBUG: Sent %lld bytes\n", sent);
	  }
	}

	if (readMode) {
	  vsib_fileno = open(vsib_device, O_WRONLY);
	} else {
	  vsib_fileno = open(vsib_device, O_RDONLY);
	}
	if (vsib_fileno == -1) {
	  snprintf(msg, 100, "Trying to re-open %s", vsib_device);
	  perror(msg);
	  exit(0);
	}
	 
	// Need to restart an integral number of integration times in the future
	// allowing for a 0.5sec lead time

	thistime = tim();
	if (roundStart) {
	  while (starttime-0.5<thistime) starttime += fileint;
	  printf("*** Will wait for %.1f sec\n", starttime-thistime);
	} else {
	  starttime=0;
	}

	// Now restart the recorder
	start_recorder(mode, skip, &starttime, fileint, 0);

	printf("DEBUG: blk changed from %d to ", blk);

	blk = (starttime-firsttime)*blocksPerSec; // Update so we record for the expected amount of time
	printf("%d  (%ld-%ld/%d\n", blk, starttime, firsttime, blocksPerSec);

	continue; // Restart loop
      }

      nwrite=nread;  // Write it all out

      if (c_mode==COMPNONE) {
	nwrite=0; // Skip the write
      } else if (c_mode != COMPALL) 
	compress_data(p, c_mode, nread, &nwrite);

      if (invert) invert_data(p, nwrite);

      if (shuffle) shuffle_data(p, nwrite, bandwidth, mode);

      if (nbits==1) {
	convertto1bit(p, nwrite);
	nwrite /=2;
      }

      if (nbits==10) { // Sign extend
	convertto16bit(p, nwrite);
      }

      if (net2) {
	splitdata(mode, p, p2, nwrite);
	nwrite /= 2;
      }

      if (mark5b) {
	send_mark5bheader (dirs[i].fid, sock, mk5bheader, &mjd, nwrite, net, 
			   udp, donetsend, &udpbufsize, p, framepersec);
      }

      if (net) {
	if (donetsend) {
	  if (mark5b&&udp) 
	    status = udpsend(sock, udp, memblk, &udpbufsize, &udpsequence, ipd);
	  else {
	    status = netsend(sock, p, nwrite);
	    if (net2) {
	      if (status) exit(1);
	      status = netsend(sock2, p2, nwrite);
	    }
	  }
	  if (status) exit(1);
	} else {
	  //printf("Skip %d kB\n", nwrite/1024);
	}
      } else {
	assert( (write(dirs[i].fid, p, nwrite)) == (ssize_t)nwrite );
      }
    }
    TIMESTOP(i);  /* accumulates close--open-write in worst case */
    TIMESTOP(ndirs);  /* call-to-call */

    TIMESTART(i);
    TIMESTART(ndirs);
    dirs[i].nowBlocks++;

    /* Time to switch to another file? */
    if (dirs[i].nowBlocks >= dirs[i].blocksPerFile) {
      /* Close the ready file before opening a new. */
      if (!net) assert( (close(dirs[i].fid) != -1) );

      starttime += fileint;

      currenttime = now();

    }
    blk++;
  }  /* while (blk<totalblks) */
  TIMESTOP(ndirs);
  totalDur = tim() - recordStart;

  /* Wait for the data to get out of VSIB. */
  if (readMode) {
    unsigned long b;

    if (!vsib_started) {
      printf("all written to bigbuf, starting vsib...\n");
      /* xxx: reload PLX FIFOs */
      vsib_ioctl(VSIB_RESET_DMA, /*skip_clear=>*/1);
      start_VSIB(mode, skip);
      usleep(2000000);  /* xxx */
    }

    vsib_ioctl(VSIB_GET_BYTES_IN_BIGBUF, (unsigned long)&b);
    while (b > 0) {
      printf("Waiting for ring buffer to empty (%lu bytes, sl=%d)\n",
	     b, usleeps);
      /* Still bytes in bigbuf waiting for transmit. */
      usleep(500000);  /* 0.5sec */
      usleeps++;
      vsib_ioctl(VSIB_GET_BYTES_IN_BIGBUF, (unsigned long)&b);
    }  /* while not bigbuf empty */

  }  /* if readMode, wait for data to get out of ring buffer */
   
  stop_recorder(&usleeps);

  /* Final report. */
  mbits = 8.0*totalblks*blksize/totalDur/1000.0/1000.0;
  if (verbose) printf(
		      "Took %f seconds, %f Mbits(dec)/s (%.1f%% of PCI33), usleeps = %d.\n",
		      totalDur,
		      mbits,
		      mbits/(33.0*4.0*8.0)*100.0,
		      usleeps
		      );
  for (i=0; i<=ndirs; i++) {
    printf("%s  max. %f seconds.\n",
	   dirs[i].nameTemplate,
	   dirs[i].maxOperTime
	   );
  }  /* for each directory */

  /* Remove shared memory to mark that 'wr/rd' is no more running. */
  if ((shId != -1) && (sh != (ptSh)-1) && (sh != NULL)) {
    assert( shmctl(shId, IPC_RMID, NULL) == 0 );
    assert( shmdt(sh) == 0 );
  }  // if shared memory was allocated

  free(vsib_device);
  free(dirs);
  if (check_stats) {
    free(stats);
  }

  return(EXIT_SUCCESS);
}  /* main */

void start_recorder (int mode, int skip, time_t *starttime, float fileint, 
		     int roundStart) {
  struct timeval current;
  struct tm *date;
  char datestr[100];
  
  if (*starttime!=0) {
    waituntil((double)*starttime-0.5, fileint);

    gettimeofday(&current, NULL);
  } else if (roundStart) { /* Want to start on a round integraton time, but no
			      starttime was passed */
    double now, nint, thisstart, dUT0;
    
    now = tim();

    /* Round the current time up to the nearest integration boundary. Add 0.5 sec
       as we need to start with 0.5 sec early */
    
    dUT0 = fmod(now+0.5, 86400.0);
    nint = ceil(dUT0/fileint);
    thisstart = now - dUT0 + nint*fileint;
    
    waituntil(thisstart, fileint);
    
    gettimeofday(&current, NULL);
    
  } else {
    // Delay the record start to a range of (0.3 -> 0.8) seconds
    // This ensures the program knows what second it is starting on.
    
    gettimeofday(&current, NULL);
    while(current.tv_usec < MINSTART_USEC || current.tv_usec > MAXSTART_USEC){
      usleep(100);
      gettimeofday(&current, NULL);
      
    }
  }
  
  *starttime = current.tv_sec + 1;
  
  date = gmtime(starttime); // will start on the next second
  strftime(datestr, 99, "%Y %B %e %H:%M:%S UTC", date);
  //printf("\nStart will occur at: %s [%.2f]\n\n", datestr, 
  //	  current.tv_usec/1.0e6);
  
  start_VSIB(mode, skip);
}


/* The vsib device file seems to have to be closed before it can be used
   again. This could be added to this routine */

void stop_recorder (int *usleeps) {
  unsigned long b;
  
  /* Stop the board, first DMA, and when the last descriptor */
  /* has been transferred, then write stop to board command register. */
  
  vsib_ioctl(VSIB_DELAYED_STOP_DMA, 0);
  
  vsib_ioctl(VSIB_IS_DMA_DONE, (unsigned long)&b);
  while (!b) {
    printf("Waiting for last DMA descriptor (sl=%d)\n",
	   *usleeps);
    usleep(100000);
    (*usleeps)++;
    vsib_ioctl(VSIB_IS_DMA_DONE, (unsigned long)&b);
  }
  
  vsib_ioctl(VSIB_SET_MODE, VSIB_MODE_STOP);
}

double tim(void) {
  struct timeval tv;
  double t;

  assert( gettimeofday(&tv, NULL) == 0 );
  t = (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;
  return t;
}  /* tim */

void
start_VSIB(int mode, int skip) {
  vsib_ioctl(VSIB_SET_MODE,
	     (VSIB_MODE_MODE(mode)
	      | VSIB_MODE_RUN
	      | (0 ? VSIB_MODE_GIGABIT : 0)
	      | VSIB_MODE_EMBED_1PPS_MARKERS
	      | (skip & 0x0000ffff))
	     );
}  /* start_VSIB */

time_t now () {
  struct timeval current;
  
  gettimeofday(&current, NULL);
  if(current.tv_usec/1000000.0 > 0.5)
    return current.tv_sec + 1;
  else
    return current.tv_sec;
}

void waituntil (double starttime, float fileint) {
   double now, secfract;
   char datestr[100];
   time_t time;
   struct tm *date;

   now = tim();

   /* We cannot be sure what timezone gettimeofday will return in
      We want GMT. The following should give us the offset in seconds */ 

   if (now > starttime) {
     time = starttime;
     date = gmtime(&time); // will start on the next second
     secfract = fmod(starttime, 1.0);
     strftime(datestr, 99, "%d/%m/%Y %H:%M", date);

     printf("Clock already past start time `%s:%.1f'\n", datestr, date->tm_sec+secfract);

     while (starttime<now) starttime += fileint; // Advance to the next boundary
   } 

   while (now < starttime) {
     double stillLeft = starttime - now;
     double newSleep;
     unsigned long sleepusecs;

     /* Sleep half of the available time. */
     newSleep = stillLeft / 2;
     sleepusecs = (unsigned long)(newSleep * 1000000.0);

     //printf("%.2f seconds until start, sleeping %.2f seconds...\n", 
     //	    stillLeft, newSleep);
     usleep(sleepusecs);

     now = tim();
   }  /* while time left */

   //now = tim();
   //time = floor(now);

   //printf("Waited until  %.2f  [%s]\n", now, ctime(&time));

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

void byte_at2vlba(unsigned char *c) {
  int i;
  unsigned char new = 0;
  
  for (i=0; i<4; i++) {
    new |= (sample_at2vlba((*c>>(i*2))&0x3)<<(i*2))&(0x3<<(i*2));
  }
  *c = new;
}

void shuffle_init(float bandwidth, int mode, int bitconvert) {
  int i, n;
  unsigned char *p;

  n = 0;

  if (mode==2) {  // 16bit - 2 channel 64 MHz
    n = (1<<16)*sizeof(short);
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

  } else if (mode==3) { // 8 bit -1  channel 64 MHz
    n = 256;
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
    fprintf(stderr, "Error: Could not initialise shuffle table for mode %d\n", mode);
    return;
  }

  for (i=0; i<n; i++) {
    byte_at2vlba(p);
    p++;
  }
  return;
}

void shuffle_data (char *p, size_t n, float bandwidth, int mode) {
  int i;
  if (bandwidth==64) {
    if (mode==2) {
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
    } else if (mode==3) {
      unsigned char *c;
      c = (unsigned char*)p;
      for (i=0; i<n;i++) {
	*c = shuffle_bytetable[*c];
	c++;
      }

    } else {
      fprintf(stderr, "Cannot shuffle 64 MHz bandwidth in mode %d\n", mode);
      exit(1);
    }
  } else {
    fprintf(stderr, "Cannot shuffle bandwidth %.1f\n", bandwidth);
    exit(1);
  }

  return;
}

// This assumes VLBA (offset binary) encoding. For sign mag (AT) opposite bits are needed
void convert_init(float bandwidth) {
  int i, j;

  convert1bit_table = malloc(pow(2,16));
  if (convert1bit_table==NULL) {
    fprintf(stderr, "Failed to allocate memory for 1 bit conversion lookup table\n");
    exit(1);
  }
  if (bandwidth==64) {
    printf("Initialising 1 bit conversion assuming Sign/Magnitude\n");
    for (i=0; i<pow(2,16); i++) {
      convert1bit_table[i] = 0;
      for (j=0; j<8; j++) 
	convert1bit_table[i] |= ((i>>j)&(1<<j)); 
    }
  } else {
    for (i=0; i<pow(2,16); i++) {
      convert1bit_table[i] = 0;
      for (j=0; j<8; j++) 
	convert1bit_table[i] |= (i>>(j+1))&(1<<j); 
    }
  }
}

void convertto1bit(char *p, size_t n) {
  int i;
  unsigned short *s;
  unsigned char *c;

  if (n%2) {
    fprintf(stderr, "Can only 1 bit convert even number of bytes (%d)\n", (int)n);
    exit(1);
  }

  s = (unsigned short*)p;
  c = (unsigned char*)p;
  for (i=0; i<n/2;i++) {
    *c = convert1bit_table[*s];
    s++;
    c++;
  }
  return;
} 

void convertto16bit(char *p, size_t n) {
  int i;
  unsigned short *s;

  if (n%2) {
    fprintf(stderr, "Can only 16 bit convert even number of bytes (%d)\n", (int)n);
    exit(1);
  }

  s = (unsigned short*)p;
  for (i=0; i<n/2;i++) {
    if (*s>0) {
      *s &= 0x03FF;  // Force top 0
    } else {
      *s |= 0xFC00;
    }
    s++;
  }
  return;
} 

void invert_init(float bandwidth, int nchan) {
  int i;

  uint32_t submask;

  if (bandwidth==64.0) {
    submask = 0x1; /* 01 */
    nchan = 1;
  } else {  /* AT */
    submask = 0x3; /* 11 */ 
  }

  // Build up submask for all channels
  for (i=0; i<nchan; i++) {
    submask |= (submask&0x3) << i*2;
  }

  mask = 0;
  for (i=0; i*2*nchan*2<32; i++) {
    mask |= submask << i*2*nchan*2; // Assumes 2 bit
  }
}

void invert_data(char *p, size_t n) {
  int i;
  uint32_t *data;

  if (n%4) {
    printf("Error: Can only invert buffers with even number of bytes. Exiting\n");
    exit(1);
  }

  data = (uint32_t*)p;
  for (i=0; i<n; i+=4) {
    *data ^= mask;  
    data++;
  }
  return;
}
  
void compress_data (char *mem, compresstype c_mode, size_t nread, size_t *nwrite) {
  char *iptr,*optr;
  int j;

  if (nread%4 !=0) {
    printf("Oops, I can only compress data if I read a multiple of 4 bytes at a time\n");
    exit(1);
  }

  iptr=optr=mem;

  switch (c_mode) {

  case COMP_000X:  // Channel 1   ooox
    for (j=0; j<(int) nread/4; j++) {
      *optr =  *iptr & 0x03;
      iptr++;
      *optr |= (*iptr & 0x03)<<2;
      iptr++;
      *optr |= (*iptr & 0x03)<<4;
      iptr++;
      *optr |= (*iptr & 0x03)<<6;
      iptr++;
      optr++;
    }
    *nwrite = nread/4;
    break; 

  case COMP_00X0:  // Channel 2   ooxo
    for (j=0; j<(int) nread/4; j++) {
      *optr =  (*iptr & 0x0C)>>2;
      iptr++;
      *optr |= (*iptr & 0x0C);
      iptr++;
      *optr |= (*iptr & 0x0C)<<2;
      iptr++;
      *optr |= (*iptr & 0x0C)<<4;
      iptr++;
      optr++;
    }
    *nwrite = nread/4;
    break; 

  case COMP_00XX:   // Channel 1+2   ooxx
    for (j=0; j<(int) nread/2; j++) {
      *optr =  *iptr & 0x0F;
      iptr++;
      *optr |= (*iptr & 0x0F)<<4;
      iptr++;
      optr++;
    }
    *nwrite = nread/2;
    break;

  case COMP_0X00:  // Channel 3   oxoo
    for (j=0; j<(int) nread/4; j++) {
      *optr =  (*iptr & 0x30)>>4;
      iptr++;
      *optr |= (*iptr & 0x30)>>2;
      iptr++;
      *optr |= (*iptr & 0x30);
      iptr++;
      *optr |= (*iptr & 0x30)<<2;
      iptr++;
      optr++;
    }
    *nwrite = nread/4;
    break; 

  case COMP_0X0X:   // Channel 1+3  oxox
    for (j=0; j<(int) nread/2; j++) {
      *optr = (*iptr & 0x03) | (*iptr & 0x30)>>2;
      iptr++;
      *optr |= (*iptr & 0x03)<<4 | (*iptr & 0x30)<<2;
      iptr++;
      optr++;
    }
    *nwrite = nread/2;
    break;

  case COMP_X000:  // Channel 4   xooo
    for (j=0; j<(int) nread/4; j++) {
      *optr =  (*iptr & 0xC0)>>6;
      iptr++;
      *optr |= (*iptr & 0xC0)>>4;
      iptr++;
      *optr |= (*iptr & 0xC0)>>2;
      iptr++;
      *optr |= (*iptr & 0xC0);
      iptr++;
      optr++;
    }
    *nwrite = nread/4;
    break; 

  case COMP_X0X0:   // Channel 2+4 xoxo
    for (j=0; j<(int) nread/2; j++) {
      *optr = (*iptr & 0x0C)>>2 | (*iptr & 0xC0)>>4;
      iptr++;
      *optr |= (*iptr & 0x0C)<<2 | (*iptr & 0xC0);
      iptr++;
      optr++;
    }
    *nwrite = nread/2;
    break;

  case COMP_XX00:   // Channel 3+4   xxoo
    for (j=0; j<(int) nread/2; j++) {
      *optr =  (*iptr & 0xF0)>>4;
      iptr++;
      *optr |= (*iptr & 0xF0);
      iptr++;
      optr++;
    }
    *nwrite = nread/2;
    break;

  case COMP_0X:  // ox
  case COMP_X0:  // xo
    /* 16 bit mode, drop every 2nd byte*/

    if (c_mode==COMP_X0) iptr++;

    for (j=0; j<nread/2; j++) {
      *optr = *iptr;
      optr++;
      iptr+=2;
    }
    *nwrite = nread/2;

    break;

  case COMP_XX0000XX:
    for (j=0; j<(int) nread/2; j++) {
      *optr = *iptr & 0x0F;
      iptr++;
      *optr |= (*iptr & 0xF0);
      iptr++;
      optr++;	
    }
    *nwrite = nread/2;
    break;

  case COMP_00XXXX00:
    for (j=0; j<(int) nread/2; j++) {
      *optr = (*iptr & 0xF0)>>4;
      iptr++;
      *optr |= (*iptr & 0x0F)<<4;
      iptr++;
      optr++;	
    }
    *nwrite = nread/2;
    break;

  default:
    printf("Warning: Do not support compressing mode %d\n", c_mode);
  }
}

void splitdata(int mode, char *mem1, char *mem2, size_t n) {
  int i=0;

  if (mode==2) {
    while (i*2<n) {
      mem1[i] = mem1[i*2];
      mem2[i] = mem1[i*2+1];
      i++;
    }
  } else if (mode==3) {
    while (i*2<n) {
      mem2[i] = (mem1[i*2]>>4)&0xF;
      mem1[i] = (mem1[i*2]&0xF) ;

      mem1[i] |= (mem1[i*2+1]&0xF)<<4;
      mem2[i] |= mem1[i*2+1]&0xF0;

      i++;
    }
  } else {
    printf("Error: Cannot split data mode %d\n", mode);
  }
    return;
}


void randbuf(char *buf, ssize_t nbytes) {
  int i;

  for (i=0; i<nbytes; i++) {
    buf[i] = (char)rand();
  }  
}  

int setup_net(char *hostname, int port, int window_size, int udp, int *sock) {
  int status;
  unsigned long ip_addr;
  struct hostent     *hostptr;
  struct linger      linger = {1, 1};
  struct sockaddr_in server;    /* Socket address */

  hostptr = gethostbyname(hostname);
  if (hostptr==NULL) {
    printf("Failed to look up hostname %s\n", hostname);
    return(1);
  }

  memcpy(&ip_addr, (char *)hostptr->h_addr, sizeof(ip_addr));
  memset((char *) &server, 0, sizeof(server));
  server.sin_family = AF_INET;
  server.sin_port = htons((unsigned short)port); 
  server.sin_addr.s_addr = ip_addr;

  printf("Connecting to %s\n",inet_ntoa(server.sin_addr));


  if (udp) {
    *sock = socket(AF_INET,SOCK_DGRAM, IPPROTO_UDP); 
    if (*sock==-1) {
      perror("Failed to allocate UDP socket");
      return(1);
    }

  } else {
    *sock = socket(AF_INET, SOCK_STREAM, 0);
    if (*sock==-1) {
      perror("Failed to allocate socket");
      return(1);
    }

    /* Set the linger option so that if we need to send a message and
       close the socket, the message shouldn't get lost */
    status = setsockopt(*sock, SOL_SOCKET, SO_LINGER, (char *)&linger,
			sizeof(struct linger)); 
    if (status!=0) {
      close(*sock);
      perror("Setting socket options");
      return(1);
    }

    if (window_size>0) {
      /* Set the window size to TCP actually works */
      status = setsockopt(*sock, SOL_SOCKET, SO_SNDBUF,
			  (char *) &window_size, sizeof(window_size));
      if (status!=0) {
	close(*sock);
	perror("Setting socket options");
	return(1);
      }
      status = setsockopt(*sock, SOL_SOCKET, SO_RCVBUF,
			(char *) &window_size, sizeof(window_size));
      if (status!=0) {
	close(*sock);
	perror("Setting socket options");
	return(1);
      }
    }
  }

  status = connect(*sock, (struct sockaddr *) &server, sizeof(server));
  if (status!=0) {
    perror("Failed to connect to server");
    return(1);
  }

  return(0);
} /* Setup Net */


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

int udpsend(int sock, int datagramsize, char *buf, int *bufsize, 
	    unsigned long long *sequence, int ipd) {
  char *ptr, str[256];

  struct msghdr msg;
  struct iovec  iovect[2];
  ssize_t nsent, ntosend;
  
  msg.msg_name       = 0;
  msg.msg_namelen    = 0;
  msg.msg_iov        = &iovect[0];
  msg.msg_iovlen     = 2;
  msg.msg_control    = 0;
  msg.msg_controllen = 0;
  msg.msg_flags      = 0;
  
  ntosend = sizeof(long long) + datagramsize;

  iovect[0].iov_base = sequence;
  iovect[0].iov_len  = sizeof(long long);
  iovect[1].iov_len  = datagramsize;

  //printf("Sending: %d+%d\n", sizeof(long long), datagramsize);
  
  ptr = buf;
  while (ptr+datagramsize<buf+*bufsize) {
    iovect[1].iov_base = ptr;

    if (ipd>0) my_usleep(ipd);
    nsent = sendmsg(sock, &msg, MSG_EOR);
    if (nsent==-1) {
      sprintf(str, "Sending %d byte UDP packet: ", (int)ntosend);
      perror(str);
      return(1);
    } else if (nsent!=ntosend) {
      printf("Only sent %d of %d bytes for UDP packet\n", (int)nsent, (int)ntosend);
      return(1);
    }
    *sequence +=1;
    ptr+=datagramsize;

  }

  *bufsize = buf+*bufsize-ptr;
  memcpy(buf, ptr, *bufsize);

  return(0);
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


void accum_stats(char *p, size_t nread, unsigned long *stats, int bytespersample) {
  int j;
  unsigned char *ptr;

  ptr = (unsigned char*)p;

  // Accumulate stats
  if (bytespersample==1) { 
    for (j=0; j<nread; j++) {
      stats[*ptr]++;
      ptr++;
    }
  } else if (bytespersample==2) {
    for (j=0; j<nread; j++) {
      stats[*ptr +(j%2)*256]++;
      ptr++;
    }
  } else if (bytespersample==4) {
    for (j=0; j<nread; j++) {
      stats[*ptr +(j%4)*256]++;
      ptr++;
    }
  } else {
    // Ignore other cases - stats will be zero
  }
}

void print_stats(unsigned long stats[], int bytespersample, float bandwidth,
		  int active_stats[]) {
  int nchan, i, j, k;
  unsigned long avstats[16][4], totalcounts;

  nchan = 0;

  for (j=0;j<bytespersample*4;j++) 
    for (k=0;k<4;k++)
      avstats[j][k] = 0;
  totalcounts = 0;

  if (bytespersample==1) {
    for (j=0; j<256; j++) { /* Go through each byte distribution */
      for (k=0; k<4; k++) { /* Do each 2 bit sample separately */
	avstats[k][j>>(k*2)&0x3] += stats[j];
      }
      totalcounts += stats[j];
    }

    /* Sum channels for multisample/byte */

    nchan=4;
    if (bandwidth==64) {  /* Single channel */
      for (j=0; j<4; j++) {
	avstats[0][j] += avstats[1][j];
	avstats[0][j] += avstats[2][j];
	avstats[0][j] += avstats[3][j];
      }
      totalcounts *= 4;
      nchan=1;

    } else if (bandwidth==32) { /* Two channels */
      for (j=0; j<4; j++) {
	avstats[0][j] += avstats[1][j];
	avstats[1][j] = avstats[2][j];
	avstats[1][j] += avstats[3][j];
      }
      totalcounts *= 2;
      nchan=2;
    }
  } else if (bytespersample==2) {
    for (k=0; k<256; k++) { /* Go through each byte distribution */
      for (j=0; j<4; j++) { /* Do each 2 bit sample separately */
	avstats[j][k>>(j*2)&0x3] += stats[k];
	avstats[j+4][k>>(j*2)&0x3] += stats[256+k];
		 }
      totalcounts += stats[k] + stats[256+k];
    }
    totalcounts /= 2; /* We have counted twice */
    
    nchan=8;
    if (bandwidth==64) { // Two channels
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
      nchan = 2;
    } else if (bandwidth==32) { //Four channels
      for (k=0; k<4; k++) {
	for (j=0; j<4; j++) {
	  avstats[k][j] = avstats[k*2][j];
	  avstats[k][j] += avstats[k*2+1][j];
	}
      }
      totalcounts *= 2;
      nchan=4;
    }
  }  else if (bytespersample==4) {
    for (k=0; k<256; k++) { /* Go through each byte distribution */
      for (j=0; j<4; j++)  /* Do each 2 bit sample separately */
	for (i=0; i<4; i++)  /* The 4 sample bytes */
	  avstats[j+i*4][k>>(j*2)&0x3] += stats[256*i+k];
    
      totalcounts += stats[k] + stats[256+k];
    }
    totalcounts /= 2; /* We have counted twice */
    nchan=16;
    if (bandwidth==64) { // Four channels
      for (j=0; j<4; j++) 
	for (k=0;k<4;k++)  {
	  avstats[k][j] = avstats[k*4][j];
	  avstats[k][j] += avstats[k*4+1][j];
	  avstats[k][j] += avstats[k*4+2][j];
	  avstats[k][j] += avstats[k*4+3][j];
	}
	totalcounts *= 4;
	nchan = 4;
    } else if (bandwidth==32) { //Four channels
      for (j=0; j<4; j++) {
	for (k=0; k<8; k++) {
	  avstats[k][j] = avstats[k*2][j];
	  avstats[k][j] += avstats[k*2+1][j];
	}
      }
      totalcounts *= 2;
      nchan=8;
    }
  }

  for (k=0; k<nchan; k++) {
    if (active_stats[k]) {
      printf("Chan %d:", k);
      for (j=0; j<4; j++) {
	printf(" %6.2f", avstats[k][j]/(float)totalcounts*100);
      }
      printf("\n");
    }
  }
}

unsigned char crcdata[6];

void send_mark5bheader (int fid, int sock, u_int32_t mk5bheader[], double *mjd, 
			size_t nwrite, int net, int udp, int donetsend, 
			int *udpbufsize, char *p, int framepersec) {
  unsigned char j1, j2, j3, s1, s2, s3, s4, s5;
  unsigned short *nframe, *crc;
  int intmjd, secmjd, fracsec, status;

   // Use pointer into header for frame number and crc
   nframe = (unsigned short*)&mk5bheader[1];
   crc = (unsigned short*)&mk5bheader[3];

  if (*nframe==0) { // Start of second, set first word of VLBA time code 
    intmjd = (int)floor(*mjd);
    secmjd = (int)floor((*mjd-intmjd)*60*60*24+5e-7);  // Avoid round off errors

    //printf("DEBUG: MJD=%d\n", intmjd);
    //printf("DEBUG: SECMJD=%d\n", secmjd);
    
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
    if (udp) {
      memcpy(p-MK5B_HEADERSIZE, mk5bheader, MK5B_HEADERSIZE);
      *udpbufsize+=nwrite+MK5B_HEADERSIZE;
    } else if (donetsend) {
      status = netsend(sock, (char*)mk5bheader, MK5B_HEADERSIZE);
      if (status) exit(1);
    } 
  } else {
    assert( write(fid, mk5bheader, MK5B_HEADERSIZE) 
	    == (4*sizeof(u_int32_t)));
  }

  *nframe = (*nframe+1) % framepersec;
  if (*nframe==0) *mjd += 1.0/(60*60*24);

}

void my_usleep(double usec) {
  double now, till;
  static double last=0;

  /* The time we are sleeping till */
  till = last+usec/1.0e6;

  /* and spin for the rest of the time */
  now = tim();
  while (now<till) {
    now = tim();
  }
  last = now;
}

void kill_hup(int sig) {
  /* We may get called twice */
  if (!simulate_1pps) {
    printf("Received HUP signal - simulate 1PPS failure\n");
    simulate_1pps = 1;
    return;
  }
  signal(sig, kill_hup); /* Re-install ourselves to disable double signals */
}  
