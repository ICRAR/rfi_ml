# Makefile of vsib.c, Linux character driver for VSIbrute PCI board.

# $Log: Makefile,v $
# Revision 1.26  2009/08/31 04:14:25  phi196
# make warning cleanup
#
# Revision 1.25  2009/08/25 23:16:29  phi196
# Remove vsib.ko from defaults
#
# Revision 1.24  2008/11/25 13:34:37  phi196
# Get module from subdir
#
# Revision 1.23  2008/09/10 10:43:15  phi196
# Added init.vsib
#
# Revision 1.22  2008/09/10 10:37:45  phi196
# Changed init.d startup option
#
# Revision 1.21  2008/03/25 00:36:09  phi196
# Merged from UDP branch
#
# Revision 1.20  2007/07/06 12:51:57  phi196
# OSX changes
#
# Revision 1.19.4.1  2007/08/31 12:17:30  phi196
# Merged from head
#
# Revision 1.19  2007/01/23 05:20:32  phi196
# Added install target
#
# Revision 1.18  2006/12/06 00:37:36  phi196
# Updates for 2.6 kernel
#
#
# Revision 1.17  2006/12/05 05:54:11  phi196
# trivial fix
#
# Revision 1.16  2006/12/05 03:18:09  phi196
# Merged from net branch
#
# Revision 1.15  2006/11/14 19:38:34  phi196
# Added comment
#
# Revision 1.14.2.2  2006/07/13 07:00:21  phi196
# 2.6 kernel modules are .ko
#
# Revision 1.14.2.1  2006/06/14 23:17:04  phi196
# Changes for 2.6 kernel Makesystem
#
# Revision 1.14  2006/03/12 11:47:15  phi196
# Header support libraries
#
# Revision 1.13  2006/02/26 23:00:36  phi196
# Header updates
#
# Revision 1.12  2006/02/10 10:14:49  phi196
# Header support
#
# Revision 1.11.2.5  2006/07/17 03:38:54  phi196
# 64 bit file offsets
#
# Revision 1.11.2.4  2006/06/05 02:23:12  phi196
# Added vsib_proxy
#
# Revision 1.11.2.3  2006/05/13 14:42:20  phi196
# Added vsib_send
#
# Revision 1.11.2.2  2006/03/16 21:21:45  phi196
# Added vsib_recv
#
# Revision 1.11.2.1  2006/03/14 07:59:42  phi196
# Added header support
#
# Revision 1.11  2006/02/06 05:01:02  phi196
# Initial version - test vheader.c
#
# Revision 1.10  2005/10/30 08:43:28  phi196
# Minor cleanup
#
# Revision 1.8  2005/05/14 02:30:19  phi196
# Added tstamp.h
#
# Revision 1.7  2005/04/08 04:07:39  phi196
# Added web publishing
#
# Revision 1.6  2005/03/29 00:22:12  phi196
# Moved files out of "common" directory
#
# Revision 1.5  2005/02/15 09:56:05  phi196
# Add FTP_AREA variable
#
# Revision 1.4  2004/11/10 06:02:48  phi196
# Makefile change for Debian
#
# Revision 1.3  2004/11/10 05:50:05  phi196
# Fixed Makefile for SuSE
#
# Revision 1.2  2004/11/10 03:45:45  phi196
# Renamed to vsib_record.c
#
# Revision 1.1  2004/11/10 02:53:12  phi196
# Initial CVS version
#
# Revision 1.14  2003/03/18 10:33:50  amn
# Moved zeroing rack number LSB from 'mrg' into 'fixmk5crc'.
#
# Revision 1.13  2003/03/05 08:01:56  amn
# March 2003 Jb2/Mh Mk4/5-style Gbit test version.
#
# Revision 1.12  2002/11/29 10:58:40  amn
# Unified ft2 and ft3; made mrg seq. numbered file opening work; wr parms.
#
# Revision 1.11  2002/11/07 09:06:49  amn
# First Mk5A 1Gbit/s tests with Jodrell.
#
# Revision 1.10  2002/11/03 10:43:03  amn
# Started using 'common' files (dstart uses tstamp.c).
#
# Revision 1.9  2002/11/01 13:19:58  amn
# Just before changing to use a common tstamp.cc file.
#
# Revision 1.8  2002/09/20 07:39:15  amn
# Added mrg.c, merge/split for multi-PC files.
#
# Revision 1.7  2002/09/02 16:39:47  amn
# Test version suitable for 50MHz/2 test pattern recording, for pb/JB.
#
# Revision 1.6  2002/08/09 11:26:56  amn
# Jul-2002 first fringes Dwingeloo test version.
#
# Revision 1.5  2002/06/15 10:07:31  amn
# Added shared memory and seek support.
#
# 15/10/04 CJP Copied from AMN version 1.4 and cleaned up

ifneq ($(KERNELRELEASE),)
 obj-m	:= vsib.o
else

CFLAGS = -Wall -O2 -I.
#CFLAGS = -Wall -g -I.

CXXFLAGS = $(CFLAGS)
LDFLAGS = -lm

EXECS = vsib_record vsib_recv testvhead vsib_send vsib_proxy

all: $(EXECS)

PWD := $(shell pwd)
KERNEL := $(shell uname -r)

vsib.ko: vsib.c vsib_ioctl.h
	make -C /usr/src/linux-$(KERNEL) SUBDIRS=$(PWD) modules

vsib_record.o: vsib_record.c vsib_ioctl.h tstamp.h tstamp.c vheader.c

vsib_record: vsib_record.o tstamp.o vheader.o
	$(CC) -Wall -o $@ $^ -lm

vsib_recv: vsib_recv.o
	$(CC) -Wall -o $@ $^ -lpthread

vsib_send: vsib_send.o
	$(CC) -Wall -o $@ $^ -lpthread

vsib_proxy: vsib_proxy.o
	$(CC) -Wall -o $@ $^ -lpthread

testvhead: vheader.o testvhead.c
	$(CC) $(CFLAGS) -o $@ $^ 

install: vsib.ko
	if test ! -d  /lib/modules/$(KERNEL)/misc ; then mkdir /lib/modules/$(KERNEL)/misc ; fi
	cp kernel/$(KERNEL)/vsib.ko /lib/modules/$(KERNEL)/misc
	depmod -a
	/etc/init.d/vsib start

clean:
	rm -rf *.o *~ $(EXECS)

dist: ftp web doc

tar: vsib.tar.gz

FTP_AREA=/nfs/ftp/people/vlbi/evlbi/software/vsib
FTP_PATH=ftp://ftp.atnf.csiro.au//pub/people/vlbi/evlbi/software/vsib/
WEB_AREA=/nfs/wwwatdocs/vlbi/evlbi

VSIB_SOFTWARE= Makefile registration.h tstamp.h tstamp.c sh.h vsib.c vsib_ioctl.h vsib_record.c dstart.c vheader.c vheader.h testvhead.c vsib_recv.c vsib_send.c vsib_proxy.c init.vsib

vsib.tar.gz: $(VSIB_SOFTWARE)
	tar zcvf $@ $^

ftp: tar
	cp -f vsib.tar.gz $(VSIB_SOFTWARE) $(FTP_AREA)
#	chmod g+w $(FTP_AREA)/*

web: $(VSIB_SOFTWARE) 
	rm -f $(WEB_AREA)/software/vsib-cvs.html
	../cvs2web.pl -template ../evlbi-software.tmpl -ftppath $(FTP_PATH) \
         -path $(WEB_AREA)/software -subdir vsib $(VSIB_SOFTWARE) > \
         $(WEB_AREA)/software/vsib-cvs.html

doc: 
	cp -r doc/recorder $(WEB_AREA)

cvs:
	cvs update

.PHONY: dist clean all ftp web doc cvs install

endif

