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
#include <getopt.h>
#include "../vsib/vheader.h"

int main (int argc, char * const argv[]) {
  int file, nfile;
  vhead *header;
  char msg[255];

  header = newheader();

  for (nfile=optind; nfile<argc; nfile++) {
    resetheader(header);

    file = open(argv[nfile], OPENOPTIONS);
    if (file==-1) {
      sprintf(msg, "Failed to open input file (%s)", argv[nfile]);
      perror(msg);
      return(1);
    }
    printf("%s\n", argv[nfile]);

    readheader(header, file, NULL);

    printf("TIME %s\n", header->time);
    printf("FILETIME %s\n", header->filetime);
    printf("HEADERSIZE %d\n", header->headersize);
    printf("HEADERVERSION %d.%d\n", header->headerversion, 
	   header->headersubversion);
    printf("RECORDERVERSION %d.%d\n", header->recorderversion,  
	   header->recordersubversion);
    printf("ANTENNAID %c%c\n", header->antennaid[0], header->antennaid[1]);
    printf("ANTENNANAME %s\n", header->antennaname);
    printf("EXPERIMENTID %s\n", header->experimentid);
    printf("NUMBITS %d\n", header->numbits);
    printf("NCHAN %d\n", header->nchan);
    printf("BANDWIDTH %.2f\n", header->bandwidth);
    printf("ENCODING %s\n", enum2str(header->encoding, encodestrs));
    printf("SEQUENCE %d\n", header->sequence);

    close(file);
  }

  destroyheader(header);

  return(0);

}
