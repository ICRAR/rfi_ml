CC=gcc
#CFLAGS=-O2 -Wall
CFLAGS=-g -Wall
CLIBS=-lm -lz

# $Id: Makefile,v 1.11 2013-03-17 03:25:20 phi196 Exp $
# $Source: /epp/atapplic/cvsroot/evlbi/recorder_server/Makefile,v $

FTP_AREA=/nfs/ftp/people/vlbi/evlbi/software/recorder_server
FTP_PATH=ftp://ftp.atnf.csiro.au/pub/people/vlbi/evlbi/software/recorder_server/
WEB_AREA=/nfs/wwwatdocs/vlbi/evlbi

SOFTWARE= README INSTALL Makefile get_disk_serials.c init_recorder_server install.pl recorder_health_checker.c recorder_server.c recorder_server.h recorder_server_init.d tokstr.c tokstr.h

EXECS = recorder_server recorder_health_checker get_disk_serials recorder_server_threads

all: $(EXECS)

recorder_server: recorder_server.o tokstr.o
	$(CC) $(CFLAGS) -o recorder_server recorder_server.o tokstr.o $(CLIBS)

recorder_server_threads: recorder_server_threads.o tokstr.o
	$(CC) $(CFLAGS) -o recorder_server_threads recorder_server_threads.o tokstr.o $(CLIBS)

recorder_health_checker: recorder_health_checker.o
	$(CC) $(CFLAGS) $(CLIBS) -o recorder_health_checker recorder_health_checker.o

get_disk_serials: get_disk_serials.o
	$(CC) $(CFLAGS) $(CLIBS) -o get_disk_serials get_disk_serials.o tokstr.o

dist: ftp web doc

tar: recorder_server.tar.gz

recorder_server.tar.gz: $(SOFTWARE)
	tar zcvf $@ $^

ftp: tar
	cp -f recorder_server.tar.gz $(SOFTWARE) $(FTP_AREA)
	chmod g+w $(FTP_AREA)/*

web: $(SOFTWARE)
	rm -f $(WEB_AREA)/software/recorder-server-cvs.html
	../cvs2web.pl -template ../evlbi-software.tmpl -ftppath $(FTP_PATH) \
        -path $(WEB_AREA)/software -subdir util $(SOFTWARE) > \
        $(WEB_AREA)/software/recorder-server-cvs.html

doc:


clean:
	rm -rf *.o *~ $(EXECS)

install:
	/etc/init.d/recorder_server stop
	cp recorder_server_threads recorder_health_checker  ~/bin
	/etc/init.d/recorder_server start

.PHONY: dist clean all ftp web doc install

