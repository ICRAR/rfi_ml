/* tstamp.h  --  UTC timestamp <--> ASCII conversions. */

#ifndef _TSTAMP_H
#define _TSTAMP_H

int getDuration(const char *word, double *result);
int getDateTime(const char *word, time_t *result);
char *formatTimestamp(struct timeval *atv, char *p);
char *getUTC(char *p);
char *toTimestamp(double t, char *p);

#endif
