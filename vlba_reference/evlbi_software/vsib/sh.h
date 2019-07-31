/* sh.h -- The common shared struct across vsibcon+vsibcli. */

#include "registration.h"
RCSID(sh_h, "$Id: sh.h,v 1.2 2005/12/23 00:22:14 phi196 Exp $");

/*
 * $Log: sh.h,v $
 * Revision 1.2  2005/12/23 00:22:14  phi196
 * Removed old header shared memory stuff
 *
 * Revision 1.1  2004/11/10 02:53:12  phi196
 * Initial CVS version
 *
 * */


/* The struct :-) */

#define SH_MAXSTR 256


typedef struct sSh {
  char currentfile[SH_MAXSTR];
  char lastfile[SH_MAXSTR];
  int bandwidth;
  int nchar;
} tSh, *ptSh;


#define fourCharLong(a,b,c,d) ( ((long)a)<<24 | ((long)b)<<16 | ((long)c)<<8 | d )
