
#COMMON = -g -Wall 
COMMON = -Wall -O2
CC=gcc $(COMMON)
CXX = g++ $(COMMON)

%.o: %.cpp
	$(CXX) $(CCFLAGS) $< -c

EXECS = vsib_checker band_invert fauto fcross vsib_header lba2mk5b vsib_settime udp_recv udp_send vlbi_fake mark5_send mk5b_updatetime wait10 dfb2vdif vsib_time recoverfiles

FFTW3DIR ?= /home/vlbi/fftw
#FFTWLIBS = -I$(FFTW3DIR)/include -L$(FFTW3DIR)/lib -lfftw3f
FFTWLIBS = -lfftw3f

PGPLOTINCS = 
PGPLOTLIBS = -L/usr/X11R6/lib -lcpgplot -lpgplot -lX11 -lpng -lz
#PGPLOTINCS = -I/usr/local/pgplot 
#PGPLOTLIBS = -L/usr/local/pgplot -L/usr/X11R6/lib -lcpgplot -lpgplot -lX11 -lpng -lg2c -lz

all: $(EXECS)

vsib_checker: vsib_checker.c ../vsib/vheader.c
	$(CC) $(VHEADINCS) -o $@ $^ -lm

vsib_header: vsib_header.c ../vsib/vheader.c
	$(CC) $(VHEADINCS) -o $@ $^

vsib_time: vsib_time.c ../vsib/vheader.c
	$(CC) $(VHEADINCS) -o $@ $^

vsib_settime: vsib_settime.c ../vsib/vheader.c
	$(CC) $(VHEADINCS) -o $@ $^

lba2mk5b: lba2mk5b.c ../vsib/vheader.c
	$(CC) $(VHEADINCS) -o $@ $^ -lm

dfb2vdif: dfb2vdif.c vdif.c
	$(CC) -o $@ $^ -lm

mark5_send: mark5_send.c
	$(CC) -o $@ $^ -lm -lpthread

udp_recv: udp_recv.c 
	$(CC) -o $@ $^ -lm

mk5b_updatetime: mk5b_updatetime.c 
	$(CC) -o $@ $^ -lm	

udp_send: udp_send.c 
	$(CC) -o $@ $^ -lm

vlbi_fake: vlbi_fake.c ../vsib/vheader.c mk5blib.c vdif.c
	$(CC) -o $@ $^ -lm

band_invert: band_invert.c ../vsib/vheader.c
	$(CC) $(VHEADINCS) -o $@ $^

fauto: fauto.c ../vsib/vheader.o
	$(CC) $(PGPLOTINCS) -o $@ $^ $(FFTWLIBS) $(PGPLOTLIBS) -lm

fcross: fcross.c ../vsib/vheader.o
	$(CC) $(PGPLOTINCS) -o $@ $^ $(FFTWLIBS) $(PGPLOTLIBS) -lm

cauto: cauto.c read_head.o
	$(CC) $(PGPLOTINCS) -o $@ $^ $(PGPLOTLIBS) -lm

wait10: wait10.c
	$(CC) -o $@ $^ -lm

socket_test_client: socket_test_client.o
	$(CC)  -o $@ $^

pps: pps.c
	$(CC) -o $@ $^

hserve: hserve.o
	$(CXX) -o $@ $^ -lpthread -lm -ludt

hclient: hclient.o
	$(CXX) -o $@ $^ -lpthread -lm -ludt


FTP_AREA=/nfs/ftp/people/vlbi/evlbi/software/util
FTP_PATH=ftp://ftp.atnf.csiro.au/pub/people/vlbi/evlbi/software/util/
WEB_AREA=/nfs/wwwatdocs/vlbi/evlbi

UTIL_SOFTWARE= Makefile band_invert.c fauto.c vsib_checker.c disko.pl vsib_header.c mk5emu.pl fcross.c lba2mk5b.c m5bheader.pl vsib_settime.c udp_recv.c udp_send.c vlbi_fake.c mk5blib.c mk5blib.h dfb2vdif.c

dist: ftp web doc

tar: vsib_util.tar.gz

vsib_util.tar.gz: $(UTIL_SOFTWARE)
	tar zcvf $@ $^

ftp: tar
	cp -f vsib_util.tar.gz $(UTIL_SOFTWARE) $(FTP_AREA)
	#chmod g+w $(FTP_AREA)/*

web: $(UTIL_SOFTWARE)
	rm -f $(WEB_AREA)/software/util-cvs.html
	../cvs2web.pl -template ../evlbi-software.tmpl -ftppath $(FTP_PATH) \
        -path $(WEB_AREA)/software -subdir util $(UTIL_SOFTWARE) > \
        $(WEB_AREA)/software/util-cvs.html

doc:
	cp doc/*.html doc/*.png $(WEB_AREA)/software

clean:
	rm -rf *.o *~ $(EXECS)

.PHONY: dist clean all ftp web doc
