/* dstart.c -- Delayed start of any command. */

/*
 * $Log: dstart.c,v $
 * Revision 1.2  2010-04-01 03:14:20  phi196
 * Work with UTC and changed getDateTime usage
 *
 * Revision 1.1  2004/11/10 02:53:12  phi196
 * Initial CVS version
 *
 * Revision 1.2  2002/11/03 10:48:28  amn
 * Removed manual in-line code which is now in tstamp.c.
 *
 * Revision 1.1  2002/11/03 10:46:32  amn
 * Took tstamp.c into use.
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


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <sys/time.h>  /* gettimeofday() */
#include <time.h>  /* mktime() */
#include <unistd.h>  /* gettimeofday(), usleep() */
#include <math.h>  /* floor() */


/* Timestamp ASCII<-->double secs conversions. */
#include "tstamp.c"

double
dsecs(void) {
  struct timeval d;

  assert( gettimeofday(&d, NULL) == 0 );
  /* 64-bit double rounds off least significant microseconds
   * if seconds difference is large. */
  return ((double)d.tv_sec + (double)d.tv_usec/1000000.0);
}  /* dsecs */


int
main(int argc, char *argv[]) {
  
  time_t time;
  double starttime,now;

  if (argc < 2) {
    fprintf(stderr, "%s: usage: %s time\n", argv[0], argv[0]);
    exit(EXIT_FAILURE);
  }

  setenv("TZ", "", 1); /* Force mktime to return gmt not local time */
  tzset();

  /* Get and convert target UTC time to Unix seconds. */
  if (getDateTime(argv[1], &time)) {
    fprintf(stderr, "%s: failed to convert ISO 8601 UTC date/time `%s'\n", argv[0], argv[1]);
    exit(EXIT_FAILURE);
  }

  /* Start half a second before full UTC seconds change. */
  starttime = time-0.5;
  
  now = dsecs();
  if (now > starttime) {
    fprintf(stderr, "%s: clock already past start time `%s'\n", argv[0], argv[1]);
    exit(EXIT_FAILURE);
  }

  while (now < starttime) {
    double stillLeft = starttime - now;
    double newSleep;
    unsigned long sleepusecs;

    /* Sleep half of the available time. */
    newSleep = stillLeft / 2;
    sleepusecs = (unsigned long)(newSleep * 1000000.0);

    fprintf(stderr, "%.2f seconds until `%s', sleeping %.2f seconds...\n", stillLeft, argv[1], newSleep);
    usleep(sleepusecs);

    now = dsecs();
  }  /* while time left */

  /* Now we could start the command on rest of command line, */
  /* but for now we just exit successfully. */

  return(EXIT_SUCCESS);
}  /* main */
