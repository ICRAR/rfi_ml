#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokstr.h"

#define BUFFSIZE 1000
#define tmp_directory "/tmp"
#define hdparm_command "/sbin/hdparm"
#define e2label_command "/sbin/e2label"
#define xfslabel_command "/usr/sbin/xfs_db"

int main(int argc,char *argv[]){
  FILE *fstab=NULL,*mdadm=NULL,*hdparm=NULL,*serialout=NULL;
  char fline[BUFFSIZE],part[BUFFSIZE],*curr_delim=NULL,mline[BUFFSIZE];
  char **disks=NULL,device[BUFFSIZE],*devloc=NULL,x_command[BUFFSIZE];
  char tmp[BUFFSIZE],hline[BUFFSIZE],*serloc=NULL,diskpath[BUFFSIZE];
  char serial_numbers[BUFFSIZE];
  int n_devs=0,i;

  /* we call this routine when we want to know the serial number 
     of the disk(s) we are recording to. it tries to get the serial
     number of the currently selected volume using hdparm. it 
     checks for raid volumes automatically using fstab and mdadm.conf */

  /* pass the path to check as the first argument */
  if ((argc!=2)&&(argc!=3)&&(argc!=5)){
    printf("Usage: %s path [--labels]\n",argv[0]);
    exit(0);
  }
/*   snprintf(diskpath,BUFFSIZE," %s",argv[1]); */
  strcpy(diskpath,argv[1]);
  serial_numbers[0]='\0';
  
  /* step 1: check for the disk in fstab */
  if ((fstab=fopen("/etc/mtab","r"))==NULL){
    /* no mtab! something's wrong */
    printf("Cannot open /etc/mtab for reading!\n");
    exit(0);
  }
  while((fgets(fline,BUFFSIZE,fstab))!=NULL){
    if (fline[0]!='#'){ /* not a comment line */
      if ((strstr(fline,diskpath))!=NULL){
	/* this is the right line, get the device */
	i=0;
	while(fline[i]!='\0'){
	  if ((fline[i]==' ')||(fline[i]=='\t'))
	    fline[i]='\0';
	  i++;
	}
	strcpy(device,fline);
	/*  	fprintf(stderr,"[s/n] device = %s\n",device); */
	if ((strncmp(device,"/dev/md",7))==0){
	  /*  	  fprintf(stderr,"[s/n] is a raid device\n"); */
	  /* actually a RAID device - look for the RAID config file */
	  if ((mdadm=fopen("/etc/mdadm/mdadm.conf","r"))==NULL){
	    fclose(fstab);
	    printf("Cannot open /etc/mdadm/mdadm.conf for reading!\n");
	    exit(0);
	  }
	  /*  	  fprintf(stderr,"[s/n] found mdadm.conf\n"); */
	  while((fgets(mline,BUFFSIZE,mdadm))!=NULL){
	    if (mline[0]!='#'){
	      if ((strstr(mline,device))!=NULL){
		devloc=strstr(mline,"devices=");
		(strstr(devloc,"\n"))[0]='\0';
		/*  		fprintf(stderr,"[s/n] md %s\n",devloc); */
		curr_delim=devloc+8;
		while((tokstr(&curr_delim,",",part))!=NULL){
		  /* split the devices on commas, note we do not support
		     RAID devices made of RAID devices */
		  fprintf(stderr,"[s/n] device = %s\n",part);
		  n_devs++;
		  disks=realloc(disks,n_devs*sizeof(char*));
		  disks[n_devs-1]=malloc(BUFFSIZE*sizeof(char));
		  strcpy(disks[n_devs-1],part);
		}
	      }
	    }
	  }
	  fclose(mdadm);
	} else {
	  /* normal disk device (we assume), so just take the name
	     of the device */
	  n_devs++;
	  disks=realloc(disks,n_devs*sizeof(char*));
	  disks[n_devs-1]=malloc(BUFFSIZE*sizeof(char));
	  strcpy(disks[n_devs-1],device);
	}
      }
    }
  }
  fclose(fstab);

  if ((argc>=3)&&(strcmp(argv[2],"--labels")==0)){

    /* we just need to get the filesystem label for the disk */
    (void)snprintf(tmp,BUFFSIZE,"%s/current_serials",tmp_directory);
    if ((argc==5)&&(strcmp(argv[3],"--filesystem")==0)){
      if (strncmp(argv[4],"ext",3)==0){
	(void)snprintf(x_command,BUFFSIZE,"%s %s > %s",e2label_command,device,tmp);
      } else if (strcmp(argv[4],"xfs")==0){
	(void)snprintf(x_command,BUFFSIZE,"%s -r -c label %s | sed 's/\\(.*\\)\\\"\\(.*\\)\\\"/\\2/' > %s",xfslabel_command,device,tmp);
      }
    } else {
      printf("must specify filesystem!\n");
      exit(-1);
    }
    system(x_command);

  } else {
    
    /* step 2: get the serial numbers of the disks */
    /*   fprintf(stderr,"[s/n] number devs = %d\n",n_devs); */
    for (i=0;i<n_devs;i++){
      fprintf(stderr,"[s/n] dev = %s\n",disks[i]);
      (void)snprintf(tmp,BUFFSIZE,"%s/disk_hdparm.out",tmp_directory);
      (void)snprintf(x_command,BUFFSIZE,"%s -I %s > %s",
		     hdparm_command,disks[i],tmp);
      /*     fprintf(stderr,"[s/n] command = [%s]\n",x_command); */
      system(x_command);
      if ((hdparm=fopen(tmp,"r"))==NULL){
	/* oops, the hdparm command didn't work */
	printf("couldn't run hdparm!\n");
	exit(0);
      }
      while((fgets(hline,BUFFSIZE,hdparm))!=NULL){
	if ((serloc=strstr(hline,"Serial Number:"))!=NULL){
	  serloc+=14;
	  while(serloc[0]==' ')
	    serloc++;
	  (strstr(serloc,"\n"))[0]='\0';
	  if (strlen(serial_numbers)!=0)
	    strcat(serial_numbers," ");
	  strcat(serial_numbers,serloc);
	}
      }
      fclose(hdparm);
      (void)snprintf(x_command,BUFFSIZE,"rm -f %s",tmp);
      system(x_command);
      free(disks[i]);
    }
    free(disks);
    
    snprintf(tmp,BUFFSIZE,"%s/current_serials",tmp_directory);
    if ((serialout=fopen(tmp,"w"))==NULL){
      printf("Couldn't open %s for writing!\n",tmp);
      exit(0);
    }
    
    fprintf(serialout,"%s\n",serial_numbers);
    fclose(serialout);

  }

  exit(0);


}
