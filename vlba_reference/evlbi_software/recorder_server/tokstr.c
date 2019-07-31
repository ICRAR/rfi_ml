#include <stdio.h>
#include <string.h>

/*
 * tokstr
 * Routine for splitting strings by a delimiter
 * (replacement for the bad strtok)
 * written by Jamie Stevens 2006
 *
 * This routine works in a similar fashion to strtok.
 * It takes three args:
 *  curr_delim - a pointer to a string which contains the
 *   string to parse.
 *  delim - a string containing the delimiter to break on
 *  part - a string which, upon return, will contain the
 *   next string token. This must be sent as a fully allocated
 *   char array.
 *
 * Usage:
 * Begin by setting curr_delim to point to the beginning of
 * the string you want to parse, and then call the tokstr
 * function. Upon return, curr_delim will point to the letter
 * directly after the first delimiter character, part will
 * contain a the text between the start of the string and
 * the delimiter (although not including the delimiter), and
 * delim will not have changed. The value of curr_delim will
 * also be the return value of the function.
 * Subsequent calls to the function, using the curr_delim
 * values returned from the function, will progress through the
 * string. When no more text exists after curr_delim, the
 * function will return NULL, and curr_delim will be set to
 * NULL.
 */

char *tokstr(char **curr_delim,char *delim,char *part){
  char *next_delim;

  if (curr_delim==NULL)
    return(NULL);
  if ((next_delim=strstr(*curr_delim,delim))==NULL){
    /* copy until the end of the string */
    strcpy(part,*curr_delim);
    if (strlen(part)>0)
      *curr_delim+=strlen(part);
    else
      *curr_delim=NULL;
  } else {
    strncpy(part,*curr_delim,(next_delim-*curr_delim));
    part[(next_delim-*curr_delim)]='\0';
    *curr_delim=++next_delim;
  }
  return(*curr_delim);
}
