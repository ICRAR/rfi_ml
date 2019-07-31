#ifdef __APPLE__

#define OSX

#define OPENOPTIONS O_RDWR

#else

#define LINUX
#define _LARGEFILE_SOURCE 
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64

#define OPENOPTIONS O_RDWR|O_LARGEFILE

#endif

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include "../vsib/vheader.h"

int main (int argc, char * const argv[]) {
  int file, status;
  vhead *header;
  char msg[255];

  if (argc!=3) {
    fprintf(stderr, "Usage: vib_settime <file> <time>\n");
    exit(1);
  }

  header = newheader();

  file = open(argv[1], OPENOPTIONS);
  if (file==-1) {
    sprintf(msg, "Failed to open input file (%s)", argv[1]);
    perror(msg);
    return(1);
  }

  readheader(header, file, NULL);

  printf(" %s -> %s\n", header->time, argv[2]);

  settimestr(header, argv[2]);

  lseek(file, 0, SEEK_SET);
  status = writeheader(header, file, NULL);
  if (status==FILEWRITEERROR) {
    perror("Trying to write header\n");
  } else if (status==FILEWRITEERROR2) {
    fprintf(stderr, "Nothing written to header\n");
  } else if (status==FILEWRITEERROR1) {
    fprintf(stderr, "Too few bytes written to header\n");
  }

  close(file);

  destroyheader(header);

  return(0);

}
