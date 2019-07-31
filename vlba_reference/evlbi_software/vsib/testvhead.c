#include "vheader.h"

int main () {
  int fileno;
  //float freq[4] = {1612, 1665, 1667, 1720};
  //polarisationtype pol[4] = {L, R, X, Y};
  //sidebandtype sideband[4] = {USB, USB, LSB, LSB};
  int status;
  vhead *header;

  struct timeval current;
  struct timezone zone;
  struct tm *date;
  time_t now;

  header = newheader();
  resetheader(header);

  gettimeofday(&current, &zone);
  now = current.tv_sec + 1;
  date = gmtime(&now); // will start on the next second

  settime(header, date);
  setrecorderversion(header, 1, 0);
  setbandwidth(header, 16.0);
  setnchan(header, 4);

  status = readprofile(header, "test.profile");
  if (status) exit(1);

  fileno = creat("junktest", S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
  if (fileno==-1) {
    perror("Opening output file: ");
    goto END;
  }

  writeheader(header, fileno, NULL);

  close(fileno);

  fileno = open("junktest", O_RDONLY);
  if (fileno==-1) {
    perror("Re-opening output file: ");
    goto END;
  }

  readheader(header, fileno, NULL);

  close(fileno);

 END:

  destroyheader(header);

  return(0);

}
