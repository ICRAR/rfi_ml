/* recorder_server: a network server for the vsib_record command for Australian eVLBI */
/* Copyright (C) 2006  Jamie Stevens */

/* This program is free software; you can redistribute it and/or */
/* modify it under the terms of the GNU General Public License */
/* as published by the Free Software Foundation; either version 2 */
/* of the License, or (at your option) any later version. */

/* This program is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/* GNU General Public License for more details. */

/* You should have received a copy of the GNU General Public License */
/* along with this program; if not, write to the Free Software */
/* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA. */

#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netinet/in.h>
#include <net/if.h>
#include <time.h>
#include <signal.h>
#include <fcntl.h>
#include <dirent.h>
#include <fnmatch.h>
#include <math.h>
#include <ctype.h>
#include <netdb.h>
#include "tokstr.h" /* my own replacement for the strtok from string.h */

#include "recorder_server.h"
#include "rs_addrs_ioctl.c" /* adapated from iftop */
/*
RCS info

$Id: recorder_server.c,v 1.36 2013-06-12 21:57:59 ste616 Exp $
$Source: /epp/atapplic/cvsroot/evlbi/recorder_server/recorder_server.c,v $
*/

#define source_version "$Id: recorder_server.c,v 1.36 2013-06-12 21:57:59 ste616 Exp $"

/* safe malloc, realloc and free */
#define MALLOC(s,t) if (((s)=malloc(t))==NULL){ return(MALLOC_FAILED); }
#define MALLOCV(s,t) if (((s)=malloc(t))==NULL){ return;}
#define REALLOC(s,t) if (((s)=realloc(s,t))==NULL){ return(REALLOC_FAILED); }
#define REALLOCV(s,t) if (((s)=realloc(s,t))==NULL){ return; }
#define FREE(s) free(s); s=NULL;

/* the global log file pointer */
FILE *logfile;
int logday;

/* a pointer to the next scheduled experiment */
experiment *next_experiment;

/* a pointer to the first of the registered clients */
registered_clients *message_clients;

/* the IP that initiated the current connection */
char connected_ip[BUFFSIZE];
char connected_on_ip[BUFFSIZE];
int  listen_socket;
int  connected_socket;
int  hanging;
int  hangingout;
int  outsock;
int  is_remote_host;

/* the status of the server is held in this structure */
server_status status_server;

/* a list of the managed recording devices */
outputdisk *disk_one;
outputdisk *local_disks;
outputdisk *remote_disks;
/* the number of the disk to be allocated next */
int next_disknumber;

/* a list of the possible recorder errors */
recordererrors *vsib_errors;

/* the name of the station */
char STATION_NAME[BUFFSIZE];

void PrintLog(char *mess){
  char time_string[BUFFSIZE],*curr_delim=NULL,part[BUFFSIZE];
  thetimenow(time_string);
  if (logfile!=NULL){ /* check for safety */
    /* we split the input on newline characters so we can put the */
    /* date at the beginning of every line of the log */
    curr_delim=mess;
    while((tokstr(&curr_delim,"\n",part))!=NULL){
      fprintf(logfile,"(%s) %s\n",time_string,part);
      (void)fflush(logfile);
    }
  }
}

void Die(char *mess, int sock){
  perror(mess);
  PrintLog(mess);
  (void)close(sock);
  exit(EXIT_FAILURE);
}

void PrintStatus(char *mess){
  /* remove trailing newlines */
  if (mess[strlen(mess)-1]=='\n')
    mess[strlen(mess)-1]='\0';
  printf("%s\n",mess);
  PrintLog(mess);
}

void DebugStatus(char *mess){
  if (DEBUG_MESSAGES==YES)
    printf("%s\n",mess);
  (void)fflush(NULL);
}

/*
  The IssueWarning routine allows for easy distribution of server warning
  messages to the screen, logfile and all registered clients.
*/
void IssueWarning(char *mess){
  char warnmessage[BUFFSIZE];
  /* remove trailing newlines */
  if (mess[strlen(mess)-1]=='\n')
    mess[strlen(mess)-1]='\0';
  (void)snprintf(warnmessage,BUFFSIZE,"+++SERVER WARNING+++\n%s\n++++++++++++++++++++\n",mess);
  PrintStatus(warnmessage);
  PushStatus(warnmessage);
}

/*
  The BroadcastMessage routine allows for easy distribution of server information
  messages to the screen, logfile and all registered clients.
*/
void BroadcastMessage(char *mess){
  char broadmessage[BUFFSIZE];
  /* remove trailing newlines */
  if (mess[strlen(mess)-1]=='\n')
    mess[strlen(mess)-1]='\0';
  (void)snprintf(broadmessage,BUFFSIZE,"---SERVER INFORMATION---\n%s\n------------------------\n",mess);
  PrintStatus(broadmessage);
  PushStatus(broadmessage);
}

/*
  The HandleClient routine gets called when a connection is made
  to the socket.
*/
void HandleClient(int sock,recorder *settings) {
  char buffer[BUFFSIZE],tag[BUFFSIZE],arg[BUFFSIZE],endtag[BUFFSIZE],failmsg[BUFFSIZE],logmesg[BUFFSIZE];
  char tmp[BUFFSIZE];
  int received = -1,recvlen=0,parse_ret=0,returncode=-1,i;
  remoterecorder *cyclerecorder=NULL;
  for (i=0;i<BUFFSIZE;i++){
    buffer[i]='\0';
    tag[i]='\0';
    arg[i]='\0';
    endtag[i]='\0';
    failmsg[i]='\0';
  }

  snprintf(tmp,BUFFSIZE,"accepting data on socket %d",sock);
/*   PrintStatus(tmp); */

  /* Receive instruction from client */
  if ((received=(int)recv(sock,buffer,BUFFSIZE,0))<0){
    PrintStatus("Failed to receive initial bytes from client");
    return;
  } else if (received==0){
    /* probably a closed remote host connection, so we check for this */
    for (cyclerecorder=status_server.remote_recorders;cyclerecorder!=NULL;
	 cyclerecorder=cyclerecorder->next){
      if (sock==cyclerecorder->connectionsocket){
	RemoveRemote(cyclerecorder,REMOTEHOST_REMOVE_IMMEDIATE);
	break;
      }
    }
    return;
  }

  recvlen=(int)strlen(buffer);

  /* parse the instruction */
  /* the instruction must have the form: <tag>instruction</tag> */
  /* possible tags are: */
  /*   mesg: send a message to the server: instruction is echoed to the screen */
  /*   data: send some data to the server: instruction should be variable=value */
  /*   cmnd: send a command to the server: instruction should be the command name */
  /*   rcst: the recorder's health, no instruction */
  /*   rcsw: a warning message from the recorder: issue warning to clients */
  /*   rcss: the start time as given by the recorder: no instruction */

  (void)snprintf(logmesg,BUFFSIZE,"received: %s",buffer);
  PrintLog(logmesg);
  if ((parse_ret=sscanf(buffer,"<%4s>%[^<]</%4s>",tag,arg,endtag))!=3){
    strcpy(failmsg,MALFORMED_REQUEST_ERR);
    returncode=MALFORMED_REQUEST;
  } else if (strcmp(tag,endtag)!=0){
    strcpy(failmsg,MALFORMED_REQUEST_ERR);
    returncode=MALFORMED_REQUEST;
  } else if (strcmp(tag,"mesg")==0){
    returncode=message_handler(arg,failmsg);
  } else if (strcmp(tag,"data")==0){
/*     PrintStatus("Receiving data from client:"); */
/*     PrintStatus(arg); */
    returncode=data_handler(arg,failmsg,settings);
  } else if (strcmp(tag,"cmnd")==0){
/*     PrintStatus("Receiving command from client:"); */
/*     PrintStatus(arg); */
    returncode=command_handler(arg,failmsg,settings);
  } else if (strcmp(tag,"rcst")==0){
/*     PrintStatus("Receiving recorder health update:"); */
/*     PrintStatus(arg); */
    returncode=health_handler(arg,failmsg,HEALTH_INFO);
  } else if (strcmp(tag,"rcsw")==0){
/*     PrintStatus("Receiving recorder health warning:"); */
/*     PrintStatus(arg); */
    returncode=health_handler(arg,failmsg,HEALTH_ERROR);
  } else if (strcmp(tag,"rcss")==0){
/*     PrintStatus("Receiving recorder start time:"); */
/*     PrintStatus(arg); */
    returncode=health_handler(arg,failmsg,HEALTH_TIME);
  }

  if ((connected_socket==sock)||(connected_socket==-1)){
    /* send confirmation */
/*     if (is_remote_host==NO){ */
    send_confirmation(returncode,failmsg,sock);
/*     } */
    if ((connected_socket==sock)&&(is_remote_host==NO)){
      (void)close(sock);
    }
  }
    
}

/*
  This routine is called when our recorder health child process
  sends back data on how the recorder is going. It collates the
  information into a linked list to give an idea of the overall
  health of the recorder.
*/
int health_handler(char *data,char *failure,int type){
  int block,BIGBUF,pct,dd,mo,yyyy,hh,mm,ss,nentries;
  char filename[BUFFSIZE],PPS[BUFFSIZE],samp_stats[BUFFSIZE],ermsg[BUFFSIZE];
  char buf[BUFFSIZE];
  recorder_health *new_file=NULL,*cycle_health=NULL,*free_health=NULL;
  time_t new_start,start_diff;

  if (type==HEALTH_INFO){
    /* the string we get passed should look similar to */
    /* ddmoyyyyhhmmss|filename|block|BIGBUF|pct|PPS */
    /* where */
    /*   ddmmyyyyhhmmss is the time the file was opened */
    /*   filename is the name of the opened file */
    /*   block is the first block of the opened file */
    /*   BIGBUF is the amount of memory (in MB) left in the BIGBUF */
    /*   pct is the percentage fraction of memory left in the BIGBUF */
    /*   PPS is the status of the PPS signal */
    if ((sscanf(data,"%2d%2d%4d%2d%2d%2d|%[^|]|%d|%d|%d|%[^|]|%[^|]",
		&dd,&mo,&yyyy,&hh,&mm,&ss,
		filename,&block,&BIGBUF,&pct,PPS,samp_stats))!=12){
      strcpy(failure,HEALTH_BAD_STRING_ERR);
      return(HEALTH_BAD_STRING);
    }
    
    /* assign the values in the health structure */
    MALLOC(new_file,sizeof(recorder_health));
    /*   new_file=malloc(sizeof(recorder_health)); */
    assign_time(yyyy,mo,dd,hh,mm,ss,&(new_file->file_time));
    strcpy(new_file->file_name,filename);
    /* output the filename to the report file */
/*     if (status_server.experiment_report!=NULL){ */
/*       fprintf(status_server.experiment_report,"%s\n",filename); */
/*     } */
    new_file->file_block=block;
    new_file->BIGBUF_level=BIGBUF;
    new_file->BIGBUF_pct=pct;
    if (strcmp(PPS,"OK")==0){
      new_file->PPS_OK=YES;
      strcpy(new_file->PPS_message,"PPS signal OK");
    } else {
      new_file->PPS_OK=NO;
      strcpy(new_file->PPS_message,PPS);
      /* let everyone know something bad happened */
/*       IssueWarning(new_file->PPS_message); */
    }
    strcpy(new_file->statistics,samp_stats);
    
    /* add this entry to the list */
    new_file->next=status_server.recstatus;
    status_server.recstatus=new_file;
    
    /* check that there is no more than 100 entries in the list */
    nentries=0;
    for (cycle_health=status_server.recstatus;
	 cycle_health!=NULL;cycle_health=cycle_health->next){
      nentries++;
      if (nentries==100){
	/* last possible entry */
	free_health=cycle_health->next;
	cycle_health->next=NULL;
	free(free_health);
      }
    }
  } else if (type==HEALTH_ERROR){
    /* the string we get passed should similar to
       ddmoyyyyhhmmss|error message
       where
       ddmoyyyyhhmmss is the time the error occurred
       error message is the error message from vsib_record */
    if ((sscanf(data,"%2d%2d%4d%2d%2d%2d|%[^|]",
                &dd,&mo,&yyyy,&hh,&mm,&ss,ermsg))!=7){
      strcpy(failure,HEALTH_BAD_STRING_ERR);
      return(HEALTH_BAD_STRING);
    }
    
    /* if it's a PPS error we have to change the last file to
       have a bad PPS */
/*     if (strncmp(ermsg,"ERROR: 1PPS transition ABSENT",29)==0){ */
/*       status_server.recstatus->PPS_OK=NO; */
/*       strcpy(status_server.recstatus->PPS_message,ermsg); */
/*     } */

    /* we echo the warning straight to the user/log */
    IssueWarning(ermsg);
  } else if (type==HEALTH_TIME){
    /* the string we get passed should be similar to 
       yyyymmdd_HHMMSS 
       where 
       yyyy is the year
       mm is the month
       dd is the day
       HH is the hour
       MM is the minute 
       SS is the second */
    if ((sscanf(data,"%4d%2d%2d_%2d%2d%2d",
		&yyyy,&mo,&dd,&hh,&mm,&ss))!=6){
      strcpy(failure,HEALTH_BAD_STRING_ERR);
      return(HEALTH_BAD_STRING);
    }
    if ((status_server.is_recording==YES)&&
	(status_server.time_verified==NO)){
      /* we assign the start time of the recorder from this
	 string */
      assign_time(yyyy,mo,dd,hh,mm,ss,&new_start);
      /* get the difference between the new start time and the old 
         start time */
      start_diff=new_start-status_server.recorder_start;
      (void)snprintf(buf,BUFFSIZE,"adjusting estimated start/end time by %d secs",
		     (int)start_diff);
/*       PrintLog(buf); */
      /* then we update the end time */
      status_server.recorder_end+=start_diff;
      status_server.time_verified=YES;
    }
  }
  /* we're done */
  return(NO_ERROR);
}

/*
  This routine handles the starting and stopping of the recording,
  and the data checking routines. It will check that the recorder
  structure has all the necessary information before doing anything.
*/
int command_handler(char *data,char *failure,recorder *settings){

  /* the valid commands: */
  /*   record-start      = start recorder */
  /*   record-stop       = stop recorder */
  /*   check-latest      = check the latest file */
  /*   check-all         = check all files */
  /*   status-record     = get the recorder status */
  /*   status-server     = get the server status */
  /*   status-settings   = get the current settings */
  /*   status-ifconfig   = get static info about network interfaces */
  /*   status-network    = get dynamic info about network activity */
  /*   status-remotehosts= get a list of the remote hosts the server knows about */
  /*   status-recsettings= get the settings used to start the recorder */
  /*   status-targets    = get the list of recording targets and their settings */
  /*   client-register   = register a new client to receive warnings and status messages */
  /*   client-remove     = de-register a message client */
  /*   experiment-load   = load an experiment profile into the queue */
  /*   experiment-save   = save the current settings as an experiment profile */
  /*   experiment-unload = remove an experiment from the queue */
  /*   experiment-list   = list the experiments in the directory and in the queue */
  /*   experiment-start  = start an experiment manually */
  /*   experiment-stop   = stop an experiment manually */
  /*   receive-start     = start an eVLBI receiver process */
  /*   receive-stop      = stop an eVLBI receiver process */
  /*   go-default        = change back to the default directory */
  /*   if-clear          = forget about all known interfaces */
  /*   rectarget-make    = assign the current settings as a recording target */
  /*   rectarget-rem     = remove a recording target */
  /*   rectarget-recall  = take the settings from rectarget and make them the current settings */
  /*   disk-selbest      = make the server choose the best disk now */
  /*   disk-swap         = swap the current disk with another that the user has selected */
  /*   disk-autoswap     = swap the current disk with another that the server thinks is suitable */
  /*   disk-userlist     = return a list of disks the user has allowed to be selected automatically */
  /*   */
  /* some of these commands accept arguments, specified in the form: */
  /*   command:arg1,arg2,...,argN */
  /* note that there are a maximum of 4 accepted arguments */

  /* argument for client-register: */
  /*   port(p)  = specify the port the client is listening on (default 1081) */
  /* argument for receive-start, receive-stop: */
  /*   remote_recorder_commonname = the common name of the remote recorder we
                                    will be receiving from */
  /* argument for rectarget-make,rectarget-recall,rectarget-rem: */
  /*   target-id = the common name of the recording target */

  /* the command disk-selbest can also take a recording target argument if you
     want the best disk to be selected for one of the recording targets */
  /* the commands disk-swap & disk-autoswap can also take a recording target argument
     if you want the swap to occur on a recording target; there is also a special target
     argument ('norecord') that will prevent the disk swap from affecting any current
     recordings */
  
  
  /* the experiment-load command takes any number of arguments (at least one), each of which is */
  /* the name of an experiment in the profile directory, or "all" to load all available profiles */
  /* the same goes for the experiment-unload command */
  /* the experiment-save command optionally takes one argument, that being the name of the experiment */
  /* to save as */

  /* there are no arguments available for start-record, stop-record, status-record, status-check, client-remove */
  /* experiment-start, experiment-stop, status-settings, status-clean, clean-start */
  /* any arguments specified with these commands will be ignored, and an */
  /* error generated. the command will however still be executed. */

  char command[BUFFSIZE],args[MAXCHECKARGS][BUFFSIZE],*tokptr;
  char *receiver_arg=NULL,target_arg[BUFFSIZE];
  int nargs,i,rec_val=-1,returncode=-1,failflag=-1,verbflag=-1,lport=-1;
  int kclean,swap_options,disable_recorder_swap=NO;
  char tmp[BUFFSIZE],experiment_arg[BUFFSIZE],part[BUFFSIZE];
  network_interfaces *clear_interface=NULL;
  rectarget *cycle_targets=NULL,*specified_target=NULL;
  outputdisk *best_disk=NULL;
  remoterecorder *best_recorder=NULL;
  recorder *swap_settings=NULL,*hold_experiment=NULL;
  
  /* set up default values for some arguments */
  kclean=0;
  lport=LISTENPORT+1; /* the default port for message clients to listen on */
  target_arg[0]='\0';

  /* parse the command string */
  tokptr=data;
  if ((tokstr(&tokptr,":",part))!=NULL){ /* is there a command? */
    strcpy(command,part);

    if (strncmp(command,"experiment",10)==0){
      /* one of the experiment commands */
      /* we just pass the argument string as is */
      strcpy(experiment_arg,tokptr);
    } else if (strncmp(command,"receive",7)==0){
      /* the receiver commands */
      /* get the string after the colon */
      receiver_arg=tokptr;
    } else if ((strcmp(command,"rectarget-make")==0)||
	       (strcmp(command,"rectarget-recall")==0)||
	       (strcmp(command,"rectarget-rem")==0)){
      /* we copy what should be the only argument */
      strcpy(target_arg,tokptr);
      if (strlen(target_arg)==0){
	/* no argument! */
	strcpy(failure,RECTARG_NONE_ERR);
	return(RECTARG_NONE);
      }
    } else if (strcmp(command,"disk-selbest")==0){
      /* did we get a target argument? */
      strcpy(target_arg,tokptr);
    } else if ((strcmp(command,"disk-swap")==0)||
	       (strcmp(command,"disk-autoswap")==0)){
      /* do we have more than argument? */
      if (strstr(tokptr,",")){
	/* run tokstr again */
	tokstr(&tokptr,",",part);
	if (strcmp(part,"norecord")==0){
	  disable_recorder_swap=YES;
	  strcpy(target_arg,tokptr);
	} else {
	  strcpy(target_arg,part);
	  if (strcmp(tokptr,"norecord")==0){
	    disable_recorder_swap=YES;
	  }
	}
      } else {
	/* only one argument, must be the recording target ID */
	strcpy(target_arg,tokptr);
      }
    } else {
      nargs=0;
      failflag=0; verbflag=0;
      /* now run through the arguments */
      for (i=0;i<MAXCHECKARGS;i++)
	args[i][0]='\0'; /* initialise the argument array */
      
      while((tokstr(&tokptr,",",part))!=NULL){
	/* check that it's a valid argument */
	if (sscanf(part,"port%s",tmp)==1){ /* have they specified a port to communicate on? */
	  lport=atoi(tmp);
	}
      }
    }

    /* which command? */
    if (strcmp(command,"record-start")==0){

      rec_val=recorder_control(RECSTART,settings,failure);
      
    } else if (strcmp(command,"record-stop")==0){

      rec_val=recorder_control(RECSTOP,settings,failure);

    } else if (strcmp(command,"status-record")==0){

      rec_val=status_control(STATUS_RECORD,settings,failure);

    } else if (strcmp(command,"status-server")==0){

      rec_val=status_control(STATUS_SERVER,settings,failure);

    } else if (strcmp(command,"status-settings")==0){

      rec_val=status_control(STATUS_SETTINGS,settings,failure);

    } else if (strcmp(command,"status-recsettings")==0){

      rec_val=status_control(STATUS_RECSETTINGS,settings,failure);

    } else if (strcmp(command,"status-ifconfig")==0){

      rec_val=status_control(STATUS_IFCONFIG,settings,failure);

    } else if (strcmp(command,"status-network")==0){

      rec_val=status_control(STATUS_NETWORK,settings,failure);

    } else if (strcmp(command,"status-receiver")==0){
      
      rec_val=status_control(STATUS_RECEIVER,settings,failure);

    } else if (strcmp(command,"status-remotehosts")==0){

      rec_val=status_control(STATUS_REMOTEHOSTS,settings,failure);

    } else if (strcmp(command,"status-targets")==0){

      rec_val=status_control(STATUS_TARGETS,settings,failure);

    } else if (strcmp(command,"client-register")==0){

      RegisterClient(connected_ip,lport);
      rec_val=NO_ERROR;

    } else if (strcmp(command,"client-remove")==0){

      RemoveClient(connected_ip,lport);
      rec_val=NO_ERROR;

    } else if (strcmp(command,"experiment-load")==0){
      
      rec_val=experiment_control(EXPERIMENT_LOAD,experiment_arg,failure);

    } else if (strcmp(command,"experiment-list")==0){

      rec_val=experiment_control(EXPERIMENT_LIST,experiment_arg,failure);

    } else if (strcmp(command,"experiment-unload")==0){

      rec_val=experiment_control(EXPERIMENT_UNLOAD,experiment_arg,failure);
      
    } else if (strcmp(command,"experiment-start")==0){

      rec_val=experiment_control(EXPERIMENT_START,experiment_arg,failure);

    } else if (strcmp(command,"experiment-stop")==0){

      rec_val=experiment_control(EXPERIMENT_STOP,experiment_arg,failure);

    } else if (strcmp(command,"experiment-save")==0){
     
      rec_val=experiment_control(EXPERIMENT_SAVE,experiment_arg,failure);
      
    } else if (strcmp(command,"receive-start")==0){
      
      rec_val=receiver_control(RECEIVER_START,receiver_arg,settings,failure);

    } else if (strcmp(command,"receive-stop")==0){

      rec_val=receiver_control(RECEIVER_STOP,receiver_arg,settings,failure);
      
    } else if (strcmp(command,"go-default")==0){
      
      rec_val=directory_default(failure);

    } else if (strcmp(command,"if-clear")==0){

      while(status_server.interfaces!=NULL){
	clear_interface=status_server.interfaces;
	status_server.interfaces=status_server.interfaces->next;
	FREE(clear_interface);
      }

    } else if (strcmp(command,"rectarget-make")==0){

      rec_val=copy_recordertarget(RECORDERTARGET_MAKE,target_arg,settings,failure);

    } else if (strcmp(command,"rectarget-rem")==0){

      rec_val=copy_recordertarget(RECORDERTARGET_REMOVE,target_arg,settings,failure);

    } else if (strcmp(command,"rectarget-recall")==0){

      rec_val=copy_recordertarget(RECORDERTARGET_RECALL,target_arg,settings,failure);

    } else if (strcmp(command,"disk-selbest")==0){

      /* this may not be the best way to do this, but for this function this
         routine will do the legwork */
      rec_val=RECTARG_NOTFOUND;
      strcpy(failure,RECTARG_NOTFOUND_ERR); /* this is by default, unless later
					       it is actually found */
      if (strlen(target_arg)==0){
	/* select from the current settings */
	bestdisk(status_server.current_settings,&best_disk,&best_recorder,NULL);
	if (best_disk!=NULL){
	  /* we did actually get a response */
	  status_server.current_settings->recordingdisk=best_disk;
	  status_server.current_settings->targetrecorder=best_recorder;
	}
	rec_val=NO_ERROR;
	failure[0]='\0';
      } else {
	/* find the right settings */
	for (cycle_targets=status_server.target_list;
	     cycle_targets!=NULL;cycle_targets=cycle_targets->next){
	  if (strcmp(cycle_targets->target_identifier,target_arg)==0){
	    bestdisk(cycle_targets->recorder_settings,&best_disk,
		     &best_recorder,NULL);
	    if (best_disk!=NULL){
	      cycle_targets->recorder_settings->recordingdisk=best_disk;
	      cycle_targets->recorder_settings->targetrecorder=best_recorder;
	    }
	    rec_val=NO_ERROR;
	    failure[0]='\0';
	    break;
	  }
	}
      }

    } else if ((strcmp(command,"disk-swap")==0)||
	       (strcmp(command,"disk-autoswap")==0)){

      /* by default we swap the disk with the disk that is currently
         user-selected, and we allow for the task to control the recorder */
      swap_options=0;
      if (strcmp(command,"disk-swap")==0){
	swap_options=DISKSWAP_USESELECTION;
      }

      if (disable_recorder_swap==NO){
	if (strlen(target_arg)>0){
	  swap_options|=DISKSWAP_THREADCONTROL;
	} else {
	  swap_options|=DISKSWAP_RECORDERCONTROL;
	}
      }

      /* find the specified recording target if we have one */
      specified_target=NULL;
      if (strlen(target_arg)>0){
	for (cycle_targets=status_server.target_list;
	     cycle_targets!=NULL;cycle_targets=cycle_targets->next){
	  if (strcmp(cycle_targets->target_identifier,target_arg)==0){
	    specified_target=cycle_targets;
	    break;
	  }
	}
	if (specified_target==NULL){
	  /* couldn't find the specified recording target, so we throw
	     an error */
	  strcpy(failure,RECTARG_NOTFOUND_ERR);
	  return(RECTARG_NOTFOUND);
	}
      }

      if (specified_target!=NULL){
	swap_settings=specified_target->recorder_settings;
      } else {
	if ((status_server.is_recording==YES)&&
	    (status_server.recording_settings!=NULL)){
	  if (swap_options & DISKSWAP_USESELECTION){
	    /* make a new settings structure */
	    MALLOC(swap_settings,sizeof(recorder*));
	    /* copy the recorder settings into it */
	    copy_recorder(status_server.recording_settings,swap_settings);
	    /* now copy the disk stuff from the current settings */
	    swap_settings->recordingdisk=status_server.current_settings->recordingdisk;
	    swap_settings->targetrecorder=status_server.current_settings->targetrecorder;
	    swap_settings->auto_disk_select=status_server.current_settings->auto_disk_select;
	  } else {
	    swap_settings=status_server.recording_settings;
	  }
	} else {
	  swap_settings=status_server.current_settings;
	}
      }

      /* if we're running an experiment we'll need to temporarily swap
         the experiment settings */
      if (status_server.experiment_mode!=EXPERIMENT_MANUAL){
	hold_experiment=next_experiment->record_settings;
	next_experiment->record_settings=swap_settings;
      }
      
      rec_val=swapdisk(swap_options,swap_settings,failure);

      /* and replace the experiment settings if we need to */
      if (status_server.experiment_mode!=EXPERIMENT_MANUAL){
	next_experiment->record_settings=hold_experiment;
      }

      /* free any used memory */
      if ((swap_settings!=status_server.current_settings)&&
	  (swap_settings!=status_server.recording_settings)&&
	  ((specified_target!=NULL)&&(swap_settings!=specified_target->recorder_settings))&&
	  (swap_settings!=NULL)){
	FREE(swap_settings);
      }

    } else if (strcmp(command,"disk-userlist")==0){

      /* make a list of the disks the user has specified as selectable should
         they specify that auto disk selection be from a list of acceptable disks;
         we use the status routine for this, even though it is not a named status
         command */
      rec_val=status_control(STATUS_USERDISKS,settings,failure);
      

    } else { /* unrecognised command */
      
      strcpy(failure,UNKNOWN_COMMAND_ERR);
      rec_val=UNKNOWN_COMMAND;
      
    }
    /* return the return value */
    return(rec_val);
    
  } else { /* no good, there's nothing there */
    returncode=UNKNOWN_COMMAND;
    strcpy(failure,UNKNOWN_COMMAND_ERR);
    return(returncode);
  }
}

/*
  This routine controls the starting and stopping of the vsib_recv
  process.
*/
int receiver_control(int action,char *commonname,recorder *settings,char *failure){
  char fullpath[BUFFSIZE],buf[BUFFSIZE],madepath[BUFFSIZE],tmpmake[BUFFSIZE],*mpos;
  char part[BUFFSIZE],temp[BUFFSIZE],sys_command[BUFFSIZE],**receiver_args=NULL;
  char temp_status[BUFFSIZE],outname[BUFFSIZE];
  int dir_check,num_args=0,i,copy_args,receiver_pid=-1,status,outfile;
  int sysretval;
  struct stat check_dir;
  pid_t mypid,childstatus;
  FILE *errorlog=NULL;
  remoterecorder *cyclerecorder=NULL;

  settings->targetrecorder=NULL;
  /* we need to know which recorder we'll be getting data from so
     we can set the correct port and TCP window size. if the user has
     specified the recorder during the receive command we use it, 
     if there is only one recorder and the user hasn't specified we
     use it we use it anyway, and we throw an error otherwise */
  if (status_server.remote_recorders==NULL){
    /* there are no recorders, so we exit immediately */
    strcpy(failure,RECV_NORECORDERS_ERR);
    return(RECV_NORECORDERS);
  }
  if ((commonname==NULL)||(commonname[0]=='\0')){
    PrintStatus("receiver_control has not been given a specific recorder");
    /* the recorder hasn't been specified, so we check whether there
       is only one choice */
    if (status_server.remote_recorders->next==NULL){
      /* only one choice, use it */
      settings->targetrecorder=status_server.remote_recorders;
      PrintStatus("receiver_control has chosen the only recorder available");
    } else {
      /* can't decide */
      strcpy(failure,RECV_RECNOTSET_ERR);
      return(RECV_RECNOTSET);
    }
  } else {
    /* look for the recorder that was specified */
    for (cyclerecorder=status_server.remote_recorders;
	 cyclerecorder!=NULL;cyclerecorder=cyclerecorder->next){
      if (strcmp(commonname,cyclerecorder->commonname)==0){
	settings->targetrecorder=cyclerecorder;
      }
    }
  }
  /* final check, do we have a recorder? */
  if (settings->targetrecorder==NULL){ /* have to be getting remote data */
    strcpy(failure,RECV_EVLBI_OFF_ERR);
    return(RECV_EVLBI_OFF);
  }

  if (action==RECEIVER_START){
    if (settings->targetrecorder->receiver!=NULL){
      /* already running a receiver from this remote recorder */
      strcpy(failure,RECV_RUNNING_ERR);
      return(RECV_RUNNING);
    }

    /* check that we have all the required information */
    if (strlen(settings->directory_name)==0){ /* the directory name hasn't been set */
      strcpy(failure,DIRECTORY_NOT_SET_ERR);
      return(DIRECTORY_NOT_SET);
    }
    
    if (settings->recordingdisk==NULL){ /* the recording disk has been reset */
      strcpy(failure,RECORDDISK_NONE_ERR);
      return(RECORDDISK_NONE);
    }

    if (settings->targetrecorder->data_communication_port<=0){
      strcpy(failure,RECV_REMPORT_ERR);
      return(RECV_REMPORT);
    }

    if (settings->targetrecorder->tcp_window_size<=0){
      strcpy(failure,RECV_TCPWINDOW_ERR);
      return(RECV_TCPWINDOW);
    }

    /* make the requested directory */
    (void)snprintf(fullpath,BUFFSIZE,"%s/%s",settings->recordingdisk->diskpath,settings->directory_name);
    if ((dir_check=stat(fullpath,&check_dir))==0){ /* something with the directory name already exists */
      if (!S_ISDIR(check_dir.st_mode)){ /* the name of the directory exists as some other type of file */
	strcpy(failure,DIRECTORY_INVALID_ERR);
	return(DIRECTORY_INVALID);
      }
      /* otherwise the directory already exists, and we don't need to do anything */
    } else {
      (void)snprintf(buf,BUFFSIZE,"making output directory %s",fullpath);
/*       PrintStatus(buf); */
      /* we need to make the directory one level at a time */
      strcpy(madepath,settings->recordingdisk->diskpath);
      strcpy(tmpmake,settings->directory_name);
      mpos=tmpmake;
      while((tokstr(&mpos,"/",part))!=NULL){
	(void)snprintf(sys_command,BUFFSIZE,"mkdir %s/%s",madepath,part);
	sysretval=system(sys_command);
	strcpy(temp,madepath);
	(void)snprintf(madepath,BUFFSIZE,"%s/%s",temp,part);
      }
    }
    if (!((dir_check=stat(fullpath,&check_dir))==0)&&
	(S_ISDIR(check_dir.st_mode))){ /* make sure the directory was made correctly */
      strcpy(failure,MKDIR_FAILED_ERR);
      return(MKDIR_FAILED);
    }
    if ((dir_check=chdir(fullpath))!=0){ /* try to change into the output directory */
      strcpy(failure,CHDIR_FAILED_ERR);
      return(CHDIR_FAILED);
    } else 
      status_server.in_dir=YES;
    
    /* prepare receive command and arguments */
    MALLOC(receiver_args,sizeof(char *));
    MALLOC(receiver_args[0],BUFFSIZE*sizeof(char));
    strcpy(receiver_args[0],recv_command);
    /* make an array of the arguments */
    num_args=2; /* TCP port and window size are always given */
    num_args*=2; /* each argument is actually 2 arguments */
    num_args++; /* add the "oneshot" option */
    num_args++; /* need a NULL argument at the end */

    REALLOC(receiver_args,(num_args+1)*sizeof(char *));
    for (i=1;i<num_args;i++) {
      MALLOC(receiver_args[i],BUFFSIZE*sizeof(char));
    }
    
    receiver_args[num_args]=NULL; /* put the NULL argument on the end of the list */

    /* specify the arguments */
    copy_args=1;
    /* TCP port number */
    strcpy(receiver_args[copy_args],"-p"); copy_args++;
    (void)snprintf(receiver_args[copy_args],BUFFSIZE,"%d",
		   settings->targetrecorder->data_communication_port);
    copy_args++;
    /* TCP window size */
    strcpy(receiver_args[copy_args],"-w"); copy_args++;
    (void)snprintf(receiver_args[copy_args],BUFFSIZE,"%f",
		   settings->targetrecorder->tcp_window_size);
    copy_args++;
    strcpy(receiver_args[copy_args],"-o"); copy_args++;

    /* launch receiver */
    PrintStatus("launching receiver:");
    (void)snprintf(temp_status,BUFFSIZE,"executing command [%s]",receiver_args[0]);
    (void)snprintf(status_server.receiver_command,BUFFSIZE,"%s",receiver_args[0]);
    PrintStatus(temp_status);
    for (i=1;i<num_args;i++){
      (void)snprintf(temp_status,BUFFSIZE," argument %d: {%s}",i,receiver_args[i]);
      (void)snprintf(temp,BUFFSIZE,"%s %s",status_server.receiver_command,receiver_args[i]);
      strcpy(status_server.receiver_command,temp);
      PrintStatus(temp_status);
    }

    if ((receiver_pid=fork())==0){ /* we are the child process */
      /* get our pid */
      mypid=getpid();
      /* close the server socket */
      (void)close(connected_socket);
      (void)close(listen_socket);
      /* and close any other sockets we're listening to */
      for (cyclerecorder=status_server.remote_recorders;cyclerecorder!=NULL;
	   cyclerecorder=cyclerecorder->next){
	if (cyclerecorder->connectionsocket>0){
	  (void)close(cyclerecorder->connectionsocket);
	}
      }
      /* turn off the alarm */
      (void)alarm(0);
      (void)signal(SIGALRM,SIG_IGN);
      /* we need to change the process group id of the child */
      /* so a Ctrl-C to the parent doesn't kill the child */
      (void)setpgid(mypid,mypid); /* create a new process group */
      /* set the Ctrl-C handler back to default */
      (void)signal(SIGINT,SIG_DFL);

      /* redirect the STDOUT and STDERR of the receiver to a file */
      (void)snprintf(outname,BUFFSIZE,"%s/receiver_%d_output",tmp_directory,(int)mypid);
      if ((outfile=open(outname,O_WRONLY|O_CREAT|O_TRUNC|O_SYNC,S_IRUSR|S_IWUSR))==-1){
	PrintStatus("Cannot redirect STDOUT and STDERR!"); /* we'll still run the receiver though */
      } else {
	DebugStatus("Redirecting STDOUT and STDERR");
	dup2(outfile,STDOUT_FILENO);
	dup2(outfile,STDERR_FILENO); /* redirect both to the same file */
	close(outfile);
      }
      /* replace the child with the receiver process */
      if ((execv(recv_command,receiver_args))==-1){ /* this is where the code ends for the child process */
	PrintStatus("didn't start recorder!\n");
	_exit(EXIT_FAILURE);
      }
    } else if (receiver_pid<0){ /* couldn't fork */
      strcpy(failure,RECV_FORK_FAIL_ERR);
      /* free our memory */
      for (i=0;i<num_args+1;i++){
	free(receiver_args[i]);
      }
      free(receiver_args);
      return(RECV_FORK_FAIL);
    }
    /* if we get to here we must be the parent process */
    sleep(1); /* wait a second to see if the receiver has started successfully */
    childstatus=waitpid((pid_t)receiver_pid,&status,WNOHANG);
    (void)snprintf(outname,BUFFSIZE,"%s/receiver_%d_output",tmp_directory,receiver_pid);
    if (childstatus!=0){
      /* shouldn't have exited so quickly, assume an error */
      if ((errorlog=fopen(outname,"r"))==NULL){
	(void)snprintf(failure,BUFFSIZE,"%s: reason unknown",RECV_START_FAIL_ERR);
	return(RECV_START_FAIL);
      }
      (void)snprintf(failure,BUFFSIZE,"%s: receiver output follows\n",RECV_START_FAIL_ERR);
      while((fgets(buf,BUFFSIZE,errorlog))!=NULL){
	strcpy(temp,failure);
	(void)snprintf(failure,BUFFSIZE,"%s%s",temp,buf);
      }
      fclose(errorlog);
      /* free our memory */
      for (i=0;i<num_args+1;i++){
	free(receiver_args[i]);
      }
      free(receiver_args);
      return(RECV_START_FAIL);
    }

    /* everything is going fine so we now allocate a new receiver 
       for this remote recorder */
    MALLOC(settings->targetrecorder->receiver,sizeof(receiver_details));
    settings->targetrecorder->receiver->receiver_pid=receiver_pid;
    strcpy(settings->targetrecorder->receiver->receiver_command,
	   recv_command);
    settings->targetrecorder->receiver->recordingdisk=settings->recordingdisk;

/*     time(&status_server.receiver_start); */
    time(&(settings->targetrecorder->receiver->receiver_start));
    /* we will kill the receiver 20 seconds after the end of the experiment, since
       we can't be exactly sure when the recorder began */
/*     status_server.receiver_end=status_server.receiver_start+ */
/*       timeinseconds(settings->record_time)+20; */
    settings->targetrecorder->receiver->receiver_end=
      settings->targetrecorder->receiver->receiver_start+
      timeinseconds(settings->record_time)+20;

    BroadcastMessage("Receiver has started");
/*     status_server.is_receiving=YES; */
/*     status_server.receiver_pid=receiver_pid; */

    /* output the serial numbers of the disks we're recording to */
    (void)snprintf(temp,BUFFSIZE,"Receiving to %s, label \"%s\"",
		   settings->recordingdisk->diskpath,
		   settings->recordingdisk->disklabel);
    PrintLog(temp);
/*     (void)snprintf(buf,BUFFSIZE,"Device Serial Numbers: %s",status_server.current_serial); */
/*     PrintLog(buf); */

    /* reset warning flag */
    status_server.disk_warning_level=WARNING_NONE;

    /* free our memory */
    for (i=0;i<num_args+1;i++){
      free(receiver_args[i]);
    }
    free(receiver_args);

    return(NO_ERROR);
    
  } else if (action==RECEIVER_STOP){
    if (settings->targetrecorder->receiver==NULL){
      strcpy(failure,RECV_STOPPED_ERR);
      return(RECV_STOPPED);
    }

    (void)snprintf(buf,BUFFSIZE,"Receiver's PID = %d",
		   settings->targetrecorder->receiver->receiver_pid);
    PrintLog(buf);

    if ((kill(settings->targetrecorder->receiver->receiver_pid,SIGKILL))!=0){
      /* something went wrong! */
      strcpy(failure,RECV_STOP_FAILED_ERR);
      return(RECV_STOP_FAILED);
    }

    BroadcastMessage("Receiver has stopped");
    
    return(NO_ERROR);
  }
    
  return(NO_ERROR);

}
 
/*
  This routine tries to change into the server's default directory.
  This is useful so that the disks attached to the recorder can
  be unmounted and removed.
*/
int directory_default(char *failure){
  int dir_check=-1;

  if ((status_server.is_recording==YES)||
      (status_server.is_receiving==YES)){
    strcpy(failure,CHDEFAULT_RUN_ERR);
    return(CHDEFAULT_RUN);
  }

  if ((dir_check=chdir(default_directory))!=0){
    /* we've been unable to change into the default directory */
    strcpy(failure,CHDEFAULT_FAIL_ERR);
    return(CHDEFAULT_FAIL);
  } else 
    status_server.in_dir=NO;

  return(NO_ERROR);
}

/*
  This routine is responsible for controlling the experiment
  actions such as loading, unloading, saving and listing.
*/
int experiment_control(int action,char *argument,char *failure){
  int lerr,nfiles,i,baderr=NO_ERROR,unloaded,retval;
  char filenames[BUFFSIZE],**locations=NULL,tmp[BUFFSIZE],*tokptr,badfail[BUFFSIZE];
  char part[BUFFSIZE];
  experiment *cycle_experiment=NULL,*experiment_next=NULL,*old_experiment=NULL;

  badfail[0]='\0';


  if (action==EXPERIMENT_LIST){
    /* list the experiments on disk */
    if ((lerr=list_experiments("*",filenames,&locations,&nfiles,failure))!=NO_ERROR){
      return(lerr);
    }
    (void)snprintf(failure,BUFFSIZE,"available experiment profiles:");
    for (i=0;i<nfiles;i++){
      strcpy(tmp,failure);
      (void)snprintf(failure,BUFFSIZE,"%s\n%s",tmp,locations[i]);
    }
    /* list the experiments already loaded */
    strcpy(tmp,failure);
    (void)snprintf(failure,BUFFSIZE,"%s\n\nloaded experiment profiles:",tmp);
    if (next_experiment==NULL){
      strcpy(tmp,failure);
      (void)snprintf(failure,BUFFSIZE,"%s\n(none)",tmp);
    } else {
      for (cycle_experiment=next_experiment;cycle_experiment!=NULL;cycle_experiment=cycle_experiment->next){
	strcpy(tmp,failure);
	(void)snprintf(failure,BUFFSIZE,"%s\n%s",tmp,cycle_experiment->experiment_id);
      }
    }
    return(STATUS_COMPLETE);


  } else if (action==EXPERIMENT_LOAD){
    /* first check to see if we were asked to load all the experiments */
    if (strlen(argument)==0){ /* no arguments were sent */
      strcpy(failure,NO_PROFILE_GIVEN_ERR);
      return(NO_PROFILE_GIVEN);
    }
    if (strcmp(argument,"all")==0){
      if ((lerr=list_experiments("*",filenames,&locations,&nfiles,failure))!=NO_ERROR){
	return(lerr);
      }
      for (i=0;i<nfiles;i++){
	if ((lerr=LoadExperiment(locations[i],failure,0))!=NO_ERROR){
	  /* couldn't load a particular experiment */
	  /* we won't return, since one bad experiment shouldn't stop the */
	  /* others from being loaded */
	  baderr=lerr;
	  strcpy(tmp,badfail);
	  (void)snprintf(badfail,BUFFSIZE,"%s\n[experiment %s] %s",tmp,locations[i],failure);
	}
      }
      /* return an error if any errors were encountered during loading */
      if (baderr!=NO_ERROR){
	strcpy(failure,badfail);
	return(baderr);
      } else {
	return(NO_ERROR);
      }
    }
    /* if we're here we've been asked to load explicit experiments */
    /* find all the experiment names given as arguments */
    tokptr=argument;
    while((tokstr(&tokptr,",",part))!=NULL){
      if ((lerr=LoadExperiment(part,failure,0))!=NO_ERROR){
	/* couldn't load a particular experiment */
	/* we won't return, since one bad experiment shouldn't stop the */
	/* others from being loaded */
	baderr=lerr;
	strcpy(tmp,badfail);
	(void)snprintf(badfail,BUFFSIZE,"%s\n[experiment %s] %s",tmp,part,failure);
      }
    }
    /* return an error if any errors were encountered during loading */
    if (baderr!=NO_ERROR){
      strcpy(failure,badfail);
      return(baderr);
    } else {
      return(NO_ERROR);
    }


  } else if (action==EXPERIMENT_UNLOAD){
    if (strlen(argument)==0){ /* no arguments were sent */
      strcpy(failure,NO_PROFILE_GIVEN_ERR);
      return(NO_PROFILE_GIVEN);
    }
    if (strcmp(argument,"all")==0){
      /* want to unload all experiments */
      if (status_server.experiment_mode==EXPERIMENT_QUEUE){
	/* we're currently running an experiment, don't unload it */
	while(next_experiment->next!=NULL){
	  experiment_next=next_experiment->next->next;
	  /* remove the target */
	  copy_recordertarget(RECORDERTARGET_REMOVE,next_experiment->next->experiment_id,
			      status_server.current_settings,failure);
	  free(next_experiment->next->record_settings);
	  free(next_experiment->next);
	  next_experiment->next=experiment_next;
	}
      } else {
	while(next_experiment!=NULL){
	  experiment_next=next_experiment->next;
	  copy_recordertarget(RECORDERTARGET_REMOVE,next_experiment->experiment_id,
			      status_server.current_settings,failure);
	  free(next_experiment->record_settings);
	  free(next_experiment);
	  next_experiment=experiment_next;
	}
      }
      return(NO_ERROR);
    } else {
      /* we've been asked to unload explicit experiments */
      /* find all the experiment names given as arguments */
      tokptr=argument;
      while((tokstr(&tokptr,",",part))!=NULL){
	unloaded=0;
	old_experiment=NULL;
	for (cycle_experiment=next_experiment;cycle_experiment!=NULL;cycle_experiment=cycle_experiment->next){
	  if (strcmp(cycle_experiment->experiment_id,part)==0){
	    /* unload this experiment */
	    copy_recordertarget(RECORDERTARGET_REMOVE,cycle_experiment->experiment_id,
				status_server.current_settings,failure);
	    if (old_experiment==NULL){
	      /* removing the first experiment */
	      if (status_server.experiment_mode==EXPERIMENT_QUEUE){
		/* we're executing this experiment, don't allow unloading */
		baderr=UNLOAD_CURRENT;
		strcpy(tmp,badfail);
		(void)snprintf(badfail,BUFFSIZE,"%s\n[experiment %s] %s",tmp,part,UNLOAD_CURRENT_ERR);
		break;
	      }
	      next_experiment=cycle_experiment->next;
	    } else {
	      /* need to point over this experiment */
	      old_experiment->next=cycle_experiment->next;
	    }
	    unloaded=1; /* success */
	    free(cycle_experiment); /* free the memory */
	    break; /* can't continue this loop anyway, since cycle_experiment->next would segfault */
	  }
	  old_experiment=cycle_experiment;
	}
	if (unloaded==0){
	  /* didn't find an experiment with this name */
	  baderr=NO_UNLOAD_MATCH;
	  strcpy(tmp,badfail);
	  (void)snprintf(badfail,BUFFSIZE,"%s\n[experiment %s] %s",tmp,part,NO_UNLOAD_MATCH_ERR);
	}
      }
      if (baderr!=NO_ERROR){
	strcpy(failure,badfail);
	return(baderr);
      } else {
	return(NO_ERROR);
      }
    }


  } else if (action==EXPERIMENT_START){
    /* the user has requested that the experiment be started */
    /* this request is rejected unless automatic experiment */
    /* control is off */
    if (status_server.execute_experiment==AUTO_EXPERIMENT_YES){
      /* auto control is on */
      strcpy(failure,MANUAL_AUTO_ON_ERR);
      return(MANUAL_AUTO_ON);
    }
    /* to start the experiment, we just call ExperimentReady */
    if ((retval=ExperimentReady(READY_MANUAL,failure))!=NO_ERROR){
      return(retval);
    }
    /* the experiment started no problems */
    return(NO_ERROR);

  } else if (action==EXPERIMENT_STOP){
    /* the user has requested that the experiment be stopped */
    /* this request is rejected unless automatic experiment */
    /* control is off */
    if (status_server.execute_experiment==AUTO_EXPERIMENT_YES){
      /* auto control is on */
      strcpy(failure,MANUAL_AUTO_ON_ERR);
      return(MANUAL_AUTO_ON);
    }
    /* stopping the experiment involves only stopping the recorder */
    /* the user can then restart the experiment if they want later */
    if ((retval=recorder_control(RECSTOP,status_server.current_settings,failure))!=NO_ERROR){
      return(retval);
    }
    /* we should mark the experiment as being stopped */
    next_experiment->started=NO;
    /* the experiment was stopped properly */
    return(NO_ERROR);


  } else if (action==EXPERIMENT_SAVE){
    /* saving an experiment means writing the current settings out to */
    /* an experiment profile on disk */
    /* if no argument is given for this command, the experiment id will */
    /* be the same as the directory name, otherwise the argument will be */
    /* taken as the experiment id */
    /* first, check that no more than one argument was given */
    if (strstr(argument,",")!=NULL){
      /* more than one argument */
      strcpy(failure,EXPSAVE_TOOMANY_ERR);
      return(EXPSAVE_TOOMANY);
    }
    /* call SaveExperiment to do the work */
    if ((retval=SaveExperiment(argument,failure))!=NO_ERROR){
      return(retval);
    }
    /* experiment saved */
    return(NO_ERROR);

  } else {
    /* the action was not understood */
    strcpy(failure,BAD_PROFILE_CMD_ERR);
    return(BAD_PROFILE_CMD);
  }
}

int list_experiments(char *pattern,char *names,char ***locations,int *nfiles,char *failure){
  struct stat expdir;
  int direxp,numfiles,n,i,totallen=0;
  struct dirent **eps=NULL;

  *locations=NULL;
  if ((direxp=stat(experiment_location,&expdir))!=0){
    /* the directory with the experiments doesn't exist */
    strcpy(failure,PROFILE_DIR_BAD_ERR);
    return(PROFILE_DIR_BAD);
  } else if (!S_ISDIR(expdir.st_mode)){
    /* the specified experiment location is not actually a directory */
    strcpy(failure,PROFILE_DIR_BAD_ERR);
    return(PROFILE_DIR_BAD);
  }
  numfiles=0;
  /* get a list of all the files in the directory */
  n=scandir(experiment_location,&eps,one,alphasort);
  if (n==-1){
    strcpy(failure,PROFILE_DIR_READ_ERR);
    return(PROFILE_DIR_READ);
  } else if (n==0){
    /* there are no files */
    strcpy(failure,PROFILE_DIR_EMPTY_ERR);
    return(PROFILE_DIR_EMPTY);
  } else {
    for (i=0;i<n;i++){
      if (eps[i]->d_type==DT_REG){ /* must be a regular file */
	if ((fnmatch(pattern,eps[i]->d_name,FNM_PATHNAME|FNM_PERIOD))==0){
	  /* the filename matches the pattern */
	  numfiles++;
	  REALLOC(*locations,numfiles*sizeof(char *));
	  (*locations)[numfiles-1]=names+totallen;
	  strcpy((*locations)[numfiles-1],eps[i]->d_name);
	  totallen+=strlen(eps[i]->d_name)+1;
	}
      }
    }
  }
  *nfiles=numfiles;
  return(NO_ERROR);
  
}

static int one(const struct dirent *unused){
  return 1;
}

void settings_string(recorder *settings,char *output){
  char tmp[BUFFSIZE],buf[BUFFSIZE];
  
  strcpy(buf,output);
  (void)snprintf(output,BUFFSIZE,"%s\n Record time:      %s",
		 buf,settings->record_time);
  strcpy(buf,output);
  if (strlen(settings->record_start_date)==0)
    strcpy(tmp,"not set");
  else
    strcpy(tmp,settings->record_start_date);
  (void)snprintf(output,BUFFSIZE,"%s\n Start date:       %s",buf,tmp);
  strcpy(buf,output);
  if (strlen(settings->record_start_time)==0)
    strcpy(tmp,"not set");
  else
    strcpy(tmp,settings->record_start_time);
  (void)snprintf(output,BUFFSIZE,"%s\n Start time:       %s",buf,tmp);
  strcpy(buf,output);
  if (strlen(settings->directory_name)==0)
    strcpy(tmp,"not set");
  else
    strcpy(tmp,settings->directory_name);
  (void)snprintf(output,BUFFSIZE,"%s\n Output directory: %s",buf,tmp);
  strcpy(buf,output);
  if (strlen(settings->filename_prefix)==0)
    strcpy(tmp,"not set");
  else
    strcpy(tmp,settings->filename_prefix);
  (void)snprintf(output,BUFFSIZE,"%s\n Filename prefix:  %s",buf,tmp);
  strcpy(buf,output);
  (void)snprintf(output,BUFFSIZE,"%s\n Compression:      %s",buf,
		 settings->compression);
  strcpy(buf,output);
  (void)snprintf(output,BUFFSIZE,"%s\n VSIB Mode:        %d",buf,
		 settings->vsib_mode);
  strcpy(buf,output);
  (void)snprintf(output,BUFFSIZE,"%s\n Bandwidth:        %d",buf,
		 settings->bandwidth);
  strcpy(buf,output);
  (void)snprintf(output,BUFFSIZE,"%s\n VSIB device:      %s",buf,
		 settings->vsib_device);
  strcpy(buf,output);
  if (settings->filesize_is_time==YES)
    strcpy(tmp,"secs");
  else
    strcpy(tmp,"blks");
  (void)snprintf(output,BUFFSIZE,"%s\n File size:        %d %s",buf,
		 (int)settings->filesize_or_time,
		 tmp);
  strcpy(buf,output);
  (void)snprintf(output,BUFFSIZE,"%s\n Block size:       %d B",buf,
		 (int)settings->blocksize);
  strcpy(buf,output);
  (void)snprintf(output,BUFFSIZE,"%s\n Clock rate:       %d MHz",buf,
		 (int)settings->clockrate);
  strcpy(buf,output);
  
  if (settings->mark5b_enabled==YES){
    strcpy(tmp,"enabled");
  } else {
    strcpy(tmp,"disabled");
  }
  (void)snprintf(output,BUFFSIZE,"%s\n Mark5B recording: %s",buf,tmp);
  strcpy(buf,output);
  
  if (settings->udp_enabled>0){
    (void)snprintf(output,BUFFSIZE,"%s\n Mark5B UDP MTU: %d",buf,
		   settings->udp_enabled);
    strcpy(buf,output);
  }
  
  if (settings->targetrecorder==NULL)
    strcpy(tmp,"local");
  else
    strcpy(tmp,settings->targetrecorder->commonname);
  (void)snprintf(output,BUFFSIZE,"%s\n Target recorder:  %s",buf,tmp);
  strcpy(buf,output);
  
  if (settings->recordingdisk==NULL)
    strcpy(tmp,"not set");
  else
    strcpy(tmp,settings->recordingdisk->diskpath);
  (void)snprintf(output,BUFFSIZE,"%s\n Output disk:      %s",buf,tmp);
  strcpy(buf,output);
  if (settings->recordingdisk!=NULL){
    if (strlen(settings->recordingdisk->disklabel)==0){
      strcpy(tmp,"N/A");
    } else {
      strcpy(tmp,settings->recordingdisk->disklabel);
    }
  } else if (settings->recordingdisk==NULL) {
    strcpy(tmp,"N/A");
  }
  
  (void)snprintf(output,BUFFSIZE,"%s\n Disk label:       %s",buf,tmp);
  strcpy(buf,output);
  if (status_server.execute_experiment==AUTO_EXPERIMENT_NO)
    strcpy(tmp,"disabled");
  else if (status_server.execute_experiment==AUTO_EXPERIMENT_YES)
    strcpy(tmp,"enabled");
  else if (status_server.execute_experiment==AUTO_EXPERIMENT_STOP)
    strcpy(tmp,"stopped");
  else {
    /* oops, shouldn't have failed here, but we set it back to 
       AUTO_EXPERIMENT_NO */
    status_server.execute_experiment=AUTO_EXPERIMENT_NO;
    strcpy(tmp,"disabled");
  }
  (void)snprintf(output,BUFFSIZE,"%s\n Auto execute:     %s",buf,tmp);
  strcpy(buf,output);
  if (status_server.rounded_start==ROUNDSTART_YES)
    strcpy(tmp,"enabled");
  else if (status_server.rounded_start==ROUNDSTART_NO)
    strcpy(tmp,"disabled");
  (void)snprintf(output,BUFFSIZE,"%s\n Auto round start: %s",buf,tmp);
  strcpy(buf,output);
  if (settings->auto_disk_select==AUTODISK_ANY){
    strcpy(tmp,"any");
  } else if (settings->auto_disk_select==AUTODISK_LOCAL){
    strcpy(tmp,"local");
  } else if (settings->auto_disk_select==AUTODISK_REMOTE){
    strcpy(tmp,"remote");
  } else if (settings->auto_disk_select==AUTODISK_LIST){
    strcpy(tmp,"user-list");
  } else if (settings->auto_disk_select==AUTODISK_DISABLED){
    strcpy(tmp,"none");
  }
  (void)snprintf(output,BUFFSIZE,"%s\n Auto disk select: %s",buf,tmp);

}

/*
  This routine is responsible for controlling the status checking
  routines.
*/
int status_control(int mode,recorder *settings,char *failure){
  int cerr=-1,nacceptable=0;
  char buf[BUFFSIZE],tmp[BUFFSIZE];
  network_interfaces *cycle_interface=NULL;
  remoterecorder *cyclerecorders=NULL;
  rectarget *cycle_targets=NULL;
  outputdisk *cycledisks=NULL;

  /* what status has the user requested */
  if (mode==STATUS_RECORD){ /* recorder status */

    /* we copy the status string into the failure message buffer */
    strcpy(failure,status_server.status_recorder);
    /* we return with an error code, but this is actually a good thing here */
    /* this is done so the message handler replies to the client with the status message */
    return(STATUS_COMPLETE);

  } else if (mode==STATUS_SERVER){ /* server status */

    strcpy(failure,status_server.status_server);
    return(STATUS_COMPLETE);

  } else if (mode==STATUS_SETTINGS){ /* the current settings */
    
    /* don't need UpdateStatus to do this for us */
    (void)snprintf(failure,BUFFSIZE,"Current Settings:");
    settings_string(status_server.current_settings,failure);

    return(STATUS_COMPLETE);

  } else if (mode==STATUS_RECSETTINGS){ /* the current settings */
    
    if (status_server.is_recording==NO){
      strcpy(failure,"Not currently recording.");
      return(STATUS_COMPLETE);
    }

    /* don't need UpdateStatus to do this for us */
    (void)snprintf(failure,BUFFSIZE,"Recording Settings:");
    settings_string(status_server.recording_settings,failure);

    return(STATUS_COMPLETE);

  } else if (mode==STATUS_TARGETS){ /* settings for all the known recording targets */
    
    if (status_server.target_list==NULL){
      strcpy(failure,"No recording targets have been configured.");
      return(STATUS_COMPLETE);
    }

    (void)snprintf(failure,BUFFSIZE,"Recording targets:");
    for (cycle_targets=status_server.target_list;
	 cycle_targets!=NULL;cycle_targets=cycle_targets->next){
      strcpy(buf,failure);
      (void)snprintf(failure,BUFFSIZE,"%s\n\nTarget '%s':",
		     buf,cycle_targets->target_identifier);
      settings_string(cycle_targets->recorder_settings,failure);
    }

    return(STATUS_COMPLETE);

  } else if (mode==STATUS_IFCONFIG){ /* the static info about the available
				        network interfaces */

    cerr=interface_config(failure); /* get info about interfaces */

    /* now assemble the status string */
    /* this is the format string */
    snprintf(tmp,BUFFSIZE,"%%%ds %%%ds %%%ds %%6s",IFNAMSIZ,MAC_LENGTH,IP_LENGTH);
    snprintf(failure,BUFFSIZE,tmp,"IFACE","MAC","IP","MTU");
    snprintf(tmp,BUFFSIZE,"%%%ds %%%ds %%%ds %%6d",IFNAMSIZ,MAC_LENGTH,IP_LENGTH);
    for (cycle_interface=status_server.interfaces;
	 cycle_interface!=NULL;cycle_interface=cycle_interface->next){
      snprintf(buf,BUFFSIZE,tmp,cycle_interface->interface_name,
	       cycle_interface->MAC,cycle_interface->ip_address,
	       cycle_interface->mtu);
      strcat(failure,"\n");
      strcat(failure,buf);
    }
    return(STATUS_COMPLETE);

  } else if (mode==STATUS_NETWORK){ /* dynamic info about network activity */

    /* assemble the status string */
    failure[0]='\0';
    for (cycle_interface=status_server.interfaces;
	 cycle_interface!=NULL;cycle_interface=cycle_interface->next){
      snprintf(tmp,BUFFSIZE,"%s\n%s:",failure,
	       cycle_interface->interface_name);
      strcpy(failure,tmp);
      snprintf(tmp,BUFFSIZE,"%s\n RX: pkt=%llu err=%lu drp=%lu ovr=%lu frm=%lu byt=%llu",
	       failure,
	       cycle_interface->stats.rx_packets,cycle_interface->stats.rx_errors,
	       cycle_interface->stats.rx_dropped,cycle_interface->stats.rx_over_errors,
	       cycle_interface->stats.rx_frame_errors,cycle_interface->stats.rx_bytes);
      strcpy(failure,tmp);
      snprintf(tmp,BUFFSIZE,"%s\n TX: pkt=%llu err=%lu drp=%lu ovr=%lu car=%lu byt=%llu",
	       failure,
	       cycle_interface->stats.tx_packets,cycle_interface->stats.tx_errors,
	       cycle_interface->stats.tx_dropped,cycle_interface->stats.tx_aborted_errors,
	       cycle_interface->stats.tx_carrier_errors,cycle_interface->stats.tx_bytes);
      strcpy(failure,tmp);
    }

  } else if (mode==STATUS_RECEIVER) {

    /* we copy the status string into the failure message buffer */
    strcpy(failure,status_server.status_receiver);
    /* we return with an error code, but this is actually a good thing here */
    /* this is done so the message handler replies to the client with the status message */
    return(STATUS_COMPLETE);

  } else if (mode==STATUS_REMOTEHOSTS) {
    
    /* we go through the list of remote hosts we have been given, and 
       return a list of them to the user */
    if (status_server.remote_recorders==NULL){
      strcpy(failure,"No remote recorders have been specified.");
    } else {
      failure[0]='\0';
      for (cyclerecorders=status_server.remote_recorders;
	   cyclerecorders!=NULL;cyclerecorders=cyclerecorders->next){
	snprintf(tmp,BUFFSIZE,"Remote recorder \"%s\":\n",cyclerecorders->commonname);
	strcat(failure,tmp);
	snprintf(tmp,BUFFSIZE," Host: %s:%d\n",cyclerecorders->hostname,
		 cyclerecorders->recorderserver_port);
	strcat(failure,tmp);
	snprintf(tmp,BUFFSIZE," Port: %d\n",cyclerecorders->data_communication_port);
	strcat(failure,tmp);
      }
    }

    return(STATUS_COMPLETE);

  } else if (mode==STATUS_USERDISKS){

    /* go through all the disks we know about and list those that
       are on the user's acceptable list */
    failure[0]='\0';
    strcpy(failure,"User acceptable disk list:");
    /* local disks first */
    for (cycledisks=disk_one;cycledisks!=NULL;
	 cycledisks=cycledisks->next){
      if (cycledisks->is_acceptable==YES){
	strcpy(buf,failure);
	(void)snprintf(tmp,BUFFSIZE,"local:%s",
		       cycledisks->diskpath);
	(void)snprintf(failure,BUFFSIZE,"%s\n%s",buf,tmp);
	nacceptable++;
      }
    }
    /* now remote disks */
    for (cyclerecorders=status_server.remote_recorders;
	 cyclerecorders!=NULL;cyclerecorders=cyclerecorders->next){
      for (cycledisks=cyclerecorders->available_disks;
	   cycledisks!=NULL;cycledisks=cycledisks->next){
	if (cycledisks->is_acceptable==YES){
	  strcpy(buf,failure);
	  (void)snprintf(tmp,BUFFSIZE,"%s:%s",
			 cyclerecorders->commonname,cycledisks->diskpath);
	  (void)snprintf(failure,BUFFSIZE,"%s\n%s",buf,tmp);
	  nacceptable++;
	}
      }
    }

    if (nacceptable==0){
      /* no disks have been marked as selectable */
      strcat(failure,"\nnone");
    }

    return(STATUS_COMPLETE);

  } else { /* unrecognised request */
    strcpy(failure,STATUS_INVALID_ERR);
    return(STATUS_INVALID);
  
  }

  return(STATUS_INVALID);
}

static int is_bad_interface_name(char *i) {
    char **p;
    for (p = bad_interface_names; *p; ++p)
        if (strncmp(i, *p, strlen(*p)) == 0)
            return 1;
    return 0;
}

/*
  This routine gathers the static info about each network interface
*/
int interface_config(char *failure){
  struct if_nameindex *nameindex;
  int j=0,k;
  int have_hw_addr=0;
  unsigned char if_hw_addr[6];
  char tmp[BUFFSIZE];
  int have_ip_addr=0,mtu;
  struct in_addr if_ip_addr;
  int result;
  network_interfaces *new_interface=NULL,*cycle_interface=NULL;

  /* if there is already information then we return immediately */
  if (status_server.interfaces!=NULL)
    return(NO_ERROR);
  
  /* get the names of all the interfaces */
  nameindex=if_nameindex();

  if (nameindex==NULL){
    /* no interface names could be gathered */
    snprintf(failure,BUFFSIZE,"Could not get interface names");
    return(IFC_NONAMES);
  }

  while (nameindex[j].if_index!=0){
    /* OK we only care about ethernet interfaces, not the loopback */
    if (is_bad_interface_name(nameindex[j].if_name)){
      j++;
      continue;
    }

    /* copy the name of the interface */
    MALLOC(new_interface,sizeof(network_interfaces));
    strcpy(new_interface->interface_name,nameindex[j].if_name);
    new_interface->MAC[0]='\0';
	   
    /* get the MAC, IP address and MTU from an external routine I
       found with the program iftop, slightly modified, since the
       original did not gather MTU */
    result=get_addrs_ioctl(nameindex[j].if_name,if_hw_addr,&if_ip_addr,&mtu);
    have_hw_addr=result & 1;
    have_ip_addr=result & 2;

    if (have_hw_addr){
      /* we got the MAC address */
      for (k=0;k<6;k++){
	snprintf(tmp,BUFFSIZE,
		 "%c%02x",k ? ':' : ' ',(unsigned int)if_hw_addr[k]);
	strcat(new_interface->MAC,tmp);
      }
      new_interface->MAC[MAC_LENGTH-1]='\0';
    }
    if (have_ip_addr){
      /* we got the IP address and MTU */
      strcpy(new_interface->ip_address,inet_ntoa(if_ip_addr));
      new_interface->mtu=mtu;
    }
    
    /* add the new interface to the end of the list, unless we didn't
       get some information */
    if (have_hw_addr && have_ip_addr){
      /* all information was gathered, add it to the list */
      for (cycle_interface=status_server.interfaces;
	   cycle_interface!=NULL;cycle_interface=cycle_interface->next){
	if (cycle_interface->next==NULL){
	  /* we've found the end of the list */
	  cycle_interface->next=new_interface;
	  if (new_interface!=NULL)
	    new_interface->next=NULL;
	}
      }
      if (status_server.interfaces==NULL){
	status_server.interfaces=new_interface;
	if (new_interface!=NULL)
	  new_interface->next=NULL;
      }
    } else {
      /* don't have all info so we'll free this element */
      FREE(new_interface);
    }
    j++;
  }

  /* free our memory */
  if_freenameindex(nameindex);

  /* we're done */
  return(NO_ERROR);

}

/*
  This routine is responsible for controlling the hard disk
  recorder.
*/
int recorder_control(int command,recorder *settings,char *failure){

  FILE *errorlog=NULL;
  int ret_res=-1,dir_check=-1,recorder_pid=-1,num_args=0,i,copy_args=1;
  int year,month,day,hour,minute,second;
  int outfile,status,upstatus,curr_compression_flag,compression_valid=NO;
  int rval,sysretval,remdiskfound;
  float fileint;
  double nint;
  pid_t childstatus,mypid;
  char rec_command[BUFFSIZE],sys_command[BUFFSIZE],**recorder_args=NULL,temp_status[BUFFSIZE];
  char fullpath[BUFFSIZE],buf[BUFFSIZE],outname[BUFFSIZE],temp[BUFFSIZE],tmpmake[BUFFSIZE],madepath[BUFFSIZE],*mpos;
  char part[BUFFSIZE],experiment_report_filename[BUFFSIZE],temp_host[BUFFSIZE];
  struct stat check_dir;
  time_t time_now,start_time,end_time,dUT0,expstart,recvfinish;
  struct tm *starting_time;
  outputdisk *cycledisk=NULL,*best_disk=NULL;
  remoterecorder *cyclerecorder=NULL,*best_recorder=NULL;

  if (command==RECSTART){
    if (status_server.is_recording==YES){
      strcpy(failure,ALREADY_RECORDING_ERR);
      return(ALREADY_RECORDING);
    } else if (status_server.is_recording==NO){
      
      /* check that we have all the required information */
      if (strlen(settings->record_time)==0){ /* the recording time hasn't been set */
	strcpy(failure,RECORDING_TIME_ERR);
	return(RECORDING_TIME);
      }

      if (strlen(settings->directory_name)==0){ /* the directory name hasn't been set */
	if ((settings->targetrecorder!=NULL && settings->targetrecorder->evlbi_enabled==NO)||
	    (settings->targetrecorder==NULL)){
	  /* we don't require the directory to be set for eVLBI mode */
	  strcpy(failure,DIRECTORY_NOT_SET_ERR);
	  return(DIRECTORY_NOT_SET);
	}
      }

      if (settings->targetrecorder==NULL){
	if (settings->recordingdisk!=NULL){
	  (void)snprintf(temp,BUFFSIZE,"before: recording to local disk %s",
			 settings->recordingdisk->diskpath);
	  PrintStatus(temp);
	}
      }

      /* now select the best disk if we've been asked to do so */
      bestdisk(settings,&best_disk,&best_recorder,NULL);
      if (best_disk!=NULL){
	settings->targetrecorder=best_recorder;
	settings->recordingdisk=best_disk;
      } /* else we're actually not allowed to choose the best disk */

      if (settings->targetrecorder==NULL){
	if (settings->recordingdisk!=NULL){
	  (void)snprintf(temp,BUFFSIZE,"after: recording to local disk %s",
			 settings->recordingdisk->diskpath);
	  PrintStatus(temp);
	}
      }

      /* do a quick check to make sure that if we're recording to a 
         remote disk that the selected disk is actually associated to the
         target recorder */
      if (settings->targetrecorder!=NULL){
	remdiskfound=0;

	if (settings->targetrecorder->evlbi_enabled==YES){
          remdiskfound=1;
        } else {
          for (cycledisk=settings->targetrecorder->available_disks;
               cycledisk!=NULL;cycledisk=cycledisk->next){
            if (settings->recordingdisk==cycledisk){
              remdiskfound=1;
            }
          }
        }
	if (remdiskfound==0){
	  /* oops something has gone wrong here */
	  strcpy(failure,UNMATCHED_DISK_ERR);
	  return(UNMATCHED_DISK);
	}
      }
      /* send the directory name to the remote recorder if required, and
         also set the recording disk and recording time */
      if (settings->targetrecorder!=NULL && settings->targetrecorder->evlbi_enabled==NO){
	(void)snprintf(temp,BUFFSIZE,"<data>directory_name=%s</data>",
		       settings->directory_name);
	if (PushRemote(temp,settings->targetrecorder)!=NO_ERROR){
	  strcpy(failure,temp);
	  return(GENERAL_ERROR);
	}
	(void)snprintf(temp,BUFFSIZE,"<data>recordingdisk=%s</data>",
		       settings->recordingdisk->diskpath);
	if (PushRemote(temp,settings->targetrecorder)!=NO_ERROR){
	  strcpy(failure,temp);
	  return(GENERAL_ERROR);
	}
	/* work out how long the receiver is going to run for:
	   - figure out how long until the start of the recording (if a start time/date
	     has been given)
	   - add the recording time */
	time(&time_now);
	recvfinish=0;
	if ((strlen(settings->record_start_date)!=0)&&
	    (strlen(settings->record_start_time)!=0)){
	  sscanf(settings->record_start_date,"%4d%2d%2d",
		 &year,&month,&day);
	  sscanf(settings->record_start_time,"%2d%2d%2d",
		 &hour,&minute,&second);
	  assign_time(year,month,day,hour,minute,second,&expstart);
	  recvfinish+=expstart-time_now;
	}
	recvfinish+=timeinseconds(settings->record_time);
	fuzzy_time(recvfinish,tmpmake);
	(void)snprintf(temp,BUFFSIZE,"<data>record_time=%s</data>",tmpmake);
	if (PushRemote(temp,settings->targetrecorder)!=NO_ERROR){
	  strcpy(failure,temp);
	  return(GENERAL_ERROR);
	}
      }

      if (strlen(settings->filename_prefix)==0){ /* the filename prefix hasn't been set */
	strcpy(failure,FILENAME_NOT_SET_ERR);
	return(FILENAME_NOT_SET);
      }

      if (settings->recordingdisk==NULL){ /* the recording disk has been reset */
	if ((settings->targetrecorder!=NULL && settings->targetrecorder->evlbi_enabled==NO)||
	    (settings->targetrecorder==NULL)){
	  /* we don't require a disk for eVLBI mode */
	  strcpy(failure,RECORDDISK_NONE_ERR);
	  return(RECORDDISK_NONE);
	}
      }

      /* make the requested directory */
      if (settings->targetrecorder==NULL){
	(void)snprintf(fullpath,BUFFSIZE,"%s/%s",settings->recordingdisk->diskpath,settings->directory_name);
	if ((dir_check=stat(fullpath,&check_dir))==0){ /* something with the directory name already exists */
	  if (!S_ISDIR(check_dir.st_mode)){ /* the name of the directory exists as some other type of file */
	    strcpy(failure,DIRECTORY_INVALID_ERR);
	    return(DIRECTORY_INVALID);
	  }
	  /* otherwise the directory already exists, and we don't need to do anything */
	} else { /* we should make the directory */
	  (void)snprintf(buf,BUFFSIZE,"making output directory %s",fullpath);
	  PrintStatus(buf);
	  /* we need to make the directory one level at a time */
	  strcpy(madepath,settings->recordingdisk->diskpath);
	  strcpy(tmpmake,settings->directory_name);
	  mpos=tmpmake;
	  while((tokstr(&mpos,"/",part))!=NULL){
	    (void)snprintf(sys_command,BUFFSIZE,"mkdir %s/%s",madepath,part);
	    sysretval=system(sys_command);
	    strcpy(temp,madepath);
	    (void)snprintf(madepath,BUFFSIZE,"%s/%s",temp,part);
	  }
	}
	if (!((dir_check=stat(fullpath,&check_dir))==0)&&
	    (S_ISDIR(check_dir.st_mode))){ /* make sure the directory was made correctly */
	  strcpy(failure,MKDIR_FAILED_ERR);
	  return(MKDIR_FAILED);
	}
	if ((dir_check=chdir(fullpath))!=0){ /* try to change into the output directory */
	  strcpy(failure,CHDIR_FAILED_ERR);
	  return(CHDIR_FAILED);
	} else {
	  status_server.in_dir=YES;
	}
      } else {
	/* for eVLBI/remote recording mode we're not recording to our own disks */
	/* just change back to the default directory and set it as valid */
	if ((rval=directory_default(failure))!=NO_ERROR)
	  return(rval);
	status_server.in_dir=YES;
      }
      
      /* check that the current compression settings are valid */
      /* first assemble our compression flags */
      if (settings->vsib_mode==3){
	curr_compression_flag=COMPRESSION_MODE3;
      } else if (settings->vsib_mode==2){
	curr_compression_flag=COMPRESSION_MODE2;
      } else 
	curr_compression_flag=0;
      if (settings->bandwidth<=16){
	curr_compression_flag=curr_compression_flag |
	  COMPRESSION_BW16;
      } else if (settings->bandwidth==32){
	curr_compression_flag=curr_compression_flag |
	  COMPRESSION_BW32;
      } else if (settings->bandwidth==64){
	curr_compression_flag=curr_compression_flag |
	  COMPRESSION_BW64;
      }
      if (strlen(settings->compression)==4){
	for (i=0;i<4;i++){
	  if (settings->compression[i]=='x'){
	    switch(i){
	    case 0:
	      curr_compression_flag=curr_compression_flag | 
		COMPRESSION_CHAN1;
	      break;
	    case 1:
	      curr_compression_flag=curr_compression_flag |
		COMPRESSION_CHAN2;
	      break;
	    case 2:
	      curr_compression_flag=curr_compression_flag |
		COMPRESSION_CHAN3;
	      break;
	    case 3:
	      curr_compression_flag=curr_compression_flag |
		COMPRESSION_CHAN4;
	      break;
	    }
	  }
	}
      } else if (strlen(settings->compression)==2){
	if (strcmp(settings->compression,"xo")==0){
	  curr_compression_flag=curr_compression_flag |
	    COMPRESSION_XO;
	} else if (strcmp(settings->compression,"ox")==0){
	  curr_compression_flag=curr_compression_flag |
	    COMPRESSION_OX;
	}
      } else if (strlen(settings->compression)==8){
	if (strcmp(settings->compression,"xxooooxx")==0){
	  curr_compression_flag=curr_compression_flag |
	    COMPRESSION_XXOOOOXX;
	} else if (strcmp(settings->compression,"ooxxxxoo")==0){
	  curr_compression_flag=curr_compression_flag |
	    COMPRESSION_OOXXXXOO;
	}
      }
      /* compare our compression flag against the list of
         allowable flags */
      for (i=0;i<nvalid_compression_modes;i++){
	if (allowed_compression_modes[i]==curr_compression_flag){
	  compression_valid=YES;
	  break;
	}
      }
      if (compression_valid==NO){
	strcpy(failure,COMPRESS_INVALID_ERR);
	return(COMPRESS_INVALID);
      }

      /* prepare recording command and arguments */
      MALLOC(recorder_args,sizeof(char *));
      strcpy(rec_command,vsib_record_command);
      MALLOC(recorder_args[0],BUFFSIZE*sizeof(char));
      strcpy(recorder_args[0],rec_command);
      /* make an array of the arguments */
      num_args=2; /* recording time and output filename are always given */
      /* determine the total number of required arguments */
      /* compression */
      num_args++;
      if (strlen(settings->record_start_time)>0)
	/* start date/time */
	num_args++;
      /* VSIB mode */
      num_args++;
      /* bandwidth */
      num_args++;
      /* VSIB device */
      num_args++;
      /* file size */
      num_args++;
      /* block size */
      num_args++;
      /* clock rate */
      num_args++;
      /* ipd */
      if (settings->targetrecorder!=NULL && settings->targetrecorder->ipd>0 &&
	  settings->targetrecorder->evlbi_enabled==YES &&
	  settings->targetrecorder->udp_enabled>0)
	num_args++;

      /* eVLBI mode options */
      if (settings->targetrecorder!=NULL){
	/* check that we have all the necessary info */
	if ((strlen(settings->targetrecorder->hostname)==0)||
	    (settings->targetrecorder->data_communication_port==-1)||
	    (settings->targetrecorder->tcp_window_size==-1)){
	  strcpy(failure,EVLBI_UNKNOWN_ERR);
	  return(EVLBI_UNKNOWN);
	}
	/* remote hostname */
	num_args++;
	/* remote port */
	num_args++;
	/* TCP window size */
	num_args++;
	/* do we use UDP */
	if (settings->targetrecorder->udp_enabled>0)
	  num_args++;
      }
      /* each argument is actually 2 arguments */
      num_args*=2;
      /* do we start on a ten-second boundary */
      if (status_server.rounded_start==ROUNDSTART_YES)
	num_args++;
      /* do we use the Mark5B */
      if (settings->mark5b_enabled==YES)
	num_args++;
      /* are we recording 1 bit data */
      if (settings->onebit_enabled==YES)
	num_args++;
      /* need a NULL argument at the end */
      num_args++;

      /* allocate the space */
      (void)snprintf(temp,BUFFSIZE,"allocating space for %d arguments",num_args);
/*       PrintStatus(temp); */
      REALLOC(recorder_args,(num_args+1)*sizeof(char *));
      for (i=1;i<num_args;i++)
	MALLOC(recorder_args[i],BUFFSIZE*sizeof(char));

      recorder_args[num_args]=NULL; /* put the NULL argument on the end of the list */

      /* specify the arguments */
      copy_args=1;
      /* record time */
      strcpy(recorder_args[copy_args],"-t"); copy_args++;
      (void)snprintf(recorder_args[copy_args],BUFFSIZE,"%s",settings->record_time);
      copy_args++;
      /* prefix */
      strcpy(recorder_args[copy_args],"-o"); copy_args++;
      (void)snprintf(recorder_args[copy_args],BUFFSIZE,"%s",settings->filename_prefix);
      copy_args++;
      /* compression */
      strcpy(recorder_args[copy_args],"-c"); copy_args++;
      (void)snprintf(recorder_args[copy_args],BUFFSIZE,"%s",settings->compression);
      copy_args++;
      if (strlen(settings->record_start_time)>0){
	/* start date/time */
	strcpy(recorder_args[copy_args],"-s"); copy_args++;
	(void)snprintf(recorder_args[copy_args],BUFFSIZE,"%8sT%6s",settings->record_start_date,settings->record_start_time);
	copy_args++;
      }
      /* VSIB mode*/
      strcpy(recorder_args[copy_args],"-m"); copy_args++;
      (void)snprintf(recorder_args[copy_args],BUFFSIZE,"%d",settings->vsib_mode);
      copy_args++;
      /* bandwidth */
      strcpy(recorder_args[copy_args],"-w"); copy_args++;
      (void)snprintf(recorder_args[copy_args],BUFFSIZE,"%d",settings->bandwidth);
      copy_args++;
      /* 10 second starting */
      if (status_server.rounded_start==ROUNDSTART_YES){
	strcpy(recorder_args[copy_args],"-x"); copy_args++;
      }
      /* Mark5B operation */
      if (settings->mark5b_enabled==YES){
	strcpy(recorder_args[copy_args],"-mk5b"); copy_args++;
      }
      /* UDP operation */
      if (settings->targetrecorder!=NULL){
	if (settings->targetrecorder->udp_enabled>0 && settings->mark5b_enabled==YES){
	  strcpy(recorder_args[copy_args],"-u"); copy_args++;
	  (void)snprintf(recorder_args[copy_args],BUFFSIZE,"%d",
			 settings->targetrecorder->udp_enabled);
	  copy_args++;
	}
      }
      if (settings->onebit_enabled==YES){
	strcpy(recorder_args[copy_args],"-1bit"); copy_args++;
      }
      /* VSIB device */
      strcpy(recorder_args[copy_args],"-e"); copy_args++;
      (void)snprintf(recorder_args[copy_args],BUFFSIZE,"%s",
		     settings->vsib_device);
      copy_args++;
      /* file size */
      strcpy(recorder_args[copy_args],"-f"); copy_args++;
      if (settings->filesize_is_time==YES)
	(void)snprintf(recorder_args[copy_args],BUFFSIZE,"%ds",
		       (int)settings->filesize_or_time);
      else if (settings->filesize_is_time==NO)
	(void)snprintf(recorder_args[copy_args],BUFFSIZE,"%d",
		       (int)settings->filesize_or_time);
      copy_args++;
      /* block size */
      strcpy(recorder_args[copy_args],"-b"); copy_args++;
      (void)snprintf(recorder_args[copy_args],BUFFSIZE,"%d",
		     (int)settings->blocksize);
      copy_args++;
      /* clock rate */
      strcpy(recorder_args[copy_args],"-r"); copy_args++;
      (void)snprintf(recorder_args[copy_args],BUFFSIZE,"%d",
		     (int)settings->clockrate);
      copy_args++;
      /* ipd */
      if (settings->targetrecorder!=NULL){
	if (settings->targetrecorder->ipd>0 && 
	    settings->targetrecorder->evlbi_enabled && 
	    settings->targetrecorder->udp_enabled) {
	  strcpy(recorder_args[copy_args],"-i"); copy_args++;
	  (void)snprintf(recorder_args[copy_args],BUFFSIZE,"%d",
			 (int)settings->targetrecorder->ipd);
	  copy_args++;
	}
	/* eVLBI mode options */
	/* remote host */
	strcpy(recorder_args[copy_args],"-H"); copy_args++;
	(void)snprintf(recorder_args[copy_args],BUFFSIZE,"%s",
		       settings->targetrecorder->hostname);
	copy_args++;
	/* remote port */
	strcpy(recorder_args[copy_args],"-p"); copy_args++;
	(void)snprintf(recorder_args[copy_args],BUFFSIZE,"%d",
		       settings->targetrecorder->data_communication_port);
	copy_args++;
	/* TCP window size */
	strcpy(recorder_args[copy_args],"-W"); copy_args++;
	(void)snprintf(recorder_args[copy_args],BUFFSIZE,"%d",
		       (int)settings->targetrecorder->tcp_window_size);
	copy_args++;
      }
      (void)snprintf(temp,BUFFSIZE,"copied %d arguments",copy_args-1);
/*       PrintStatus(temp); */

      /* launch the receiver on the remote recorder if we're
         using one */
      if ((settings->targetrecorder!=NULL)&&
	  (settings->targetrecorder->evlbi_enabled==NO)){
	conjugate_host(settings->targetrecorder->commonname,temp_host);
	snprintf(temp,BUFFSIZE,"<cmnd>receive-start:%s</cmnd>",temp_host);
	if (PushRemote(temp,settings->targetrecorder)!=NO_ERROR){
	  strcpy(failure,temp);
	  return(GENERAL_ERROR);
	}
      }

      /* launch recorder */
      PrintStatus("launching recorder:");
      (void)snprintf(temp_status,BUFFSIZE,"executing command [%s]",recorder_args[0]);
      (void)snprintf(status_server.recorder_command,BUFFSIZE,"%s",recorder_args[0]);
      PrintStatus(temp_status);
      for (i=1;i<num_args;i++){
	(void)snprintf(temp_status,BUFFSIZE," argument %d: {%s}",i,recorder_args[i]);
	(void)snprintf(temp,BUFFSIZE,"%s %s",status_server.recorder_command,recorder_args[i]);
	strcpy(status_server.recorder_command,temp);
	PrintStatus(temp_status);
      }
      
      if ((recorder_pid=fork())==0){ /* are we the child process */
	/* there are a number of things to stop before running */
	/* the recorder */
	/* get our pid */
	mypid=getpid();
	/* close the server socket */
	(void)close(connected_socket);
	(void)close(listen_socket);
	/* and close any other sockets we're listening to */
	for (cyclerecorder=status_server.remote_recorders;cyclerecorder!=NULL;
	     cyclerecorder=cyclerecorder->next){
	  if (cyclerecorder->connectionsocket>0){
	    (void)close(cyclerecorder->connectionsocket);
	  }
	}
	/* turn off the alarm */
	(void)alarm(0);
	(void)signal(SIGALRM,SIG_IGN);
	/* we need to change the process group id of the child */
	/* so a Ctrl-C to the parent doesn't kill the child */
	(void)setpgid(mypid,mypid); /* create a new process group */
	/* set the Ctrl-C handler back to default */
	(void)signal(SIGINT,SIG_DFL);

	/* redirect the STDOUT and STDERR of the recorder to a file */
	(void)snprintf(outname,BUFFSIZE,"%s/recorder_%d_output",tmp_directory,(int)mypid);
	if ((outfile=open(outname,O_WRONLY|O_CREAT|O_TRUNC|O_SYNC,S_IRUSR|S_IWUSR))==-1){
	  PrintStatus("Cannot redirect STDOUT and STDERR!"); /* we'll still run the recorder though */
	} else {
	  DebugStatus("Redirecting STDOUT and STDERR");
	  dup2(outfile,STDOUT_FILENO);
	  dup2(outfile,STDERR_FILENO); /* redirect both to the same file */
	  close(outfile);
	}
	/* replace the child with the recorder process */
	if ((execv(rec_command,recorder_args))==-1){ /* this is where the code ends for the child process */
	  PrintStatus("didn't start recorder!\n");
	  _exit(EXIT_FAILURE);
	}
      } else if (recorder_pid<0){ /* couldn't fork */
	strcpy(failure,RECORD_FORK_FAIL_ERR);
      /* free our memory */
	for (i=0;i<num_args+1;i++){
	  free(recorder_args[i]);
	}
	free(recorder_args);
	return(RECORD_FORK_FAIL);
      }
      /* if we get to here we must be the parent process */
      sleep(1); /* wait a second to see if the recorder has started successfully */
      childstatus=waitpid((pid_t)recorder_pid,&status,WNOHANG);
      (void)snprintf(outname,BUFFSIZE,"%s/recorder_%d_output",tmp_directory,recorder_pid);
      if (childstatus!=0){
	/* shouldn't have exited so quickly, assume an error */
	if ((errorlog=fopen(outname,"r"))==NULL){
	  (void)snprintf(failure,BUFFSIZE,"%s: reason unknown",RECORD_START_FAIL_ERR);
	  return(RECORD_START_FAIL);
	}
	(void)snprintf(failure,BUFFSIZE,"%s: recorder output follows\n",RECORD_START_FAIL_ERR);
	while((fgets(buf,BUFFSIZE,errorlog))!=NULL){
	  strcpy(temp,failure);
	  (void)snprintf(failure,BUFFSIZE,"%s%s",temp,buf);
	}
	fclose(errorlog);
	/* free our memory */
	for (i=0;i<num_args+1;i++){
	  free(recorder_args[i]);
	}
	free(recorder_args);
	return(RECORD_START_FAIL);
      }

      time(&time_now);
      /* estimate the start time of the recorder - we'll get the
         actual start time from our health checker later */
      if (strlen(settings->record_start_time)==0){
	start_time=time_now;
      } else {
	sscanf(settings->record_start_date,"%4d%2d%2d",&year,&month,&day);
	sscanf(settings->record_start_time,"%2d%2d%2d",&hour,&minute,&second);
	assign_time(year,month,day,hour,minute,second,&start_time);
      }
      if (status_server.rounded_start==ROUNDSTART_YES){
	/* start at the nearest boundary */
	if (settings->filesize_is_time==YES)
	  fileint=settings->filesize_or_time;
	else
	  fileint=settings->filesize_or_time*
	    settings->blocksize/(float)(recordingdatarate(settings)*1E6);
	dUT0=start_time % 86400;
	nint=floor((float)dUT0/fileint);
	start_time=(start_time-dUT0)+nint*fileint;
	while(start_time<=time_now+1)
	  start_time+=fileint;
      }
      starting_time=gmtime(&start_time);
      (void)snprintf(temp,BUFFSIZE,
		     "estimate recorder start time to be %4d%02d%02d_%02d%02d%02d",
		     starting_time->tm_year+1900,starting_time->tm_mon+1,
		     starting_time->tm_mday,starting_time->tm_hour,
		     starting_time->tm_min,starting_time->tm_sec);
      status_server.time_verified=NO; /* request verification of start time */

      /* assign the experiment report filename */
      snprintf(experiment_report_filename,BUFFSIZE,
	       "%s/experiment_%s_report_%4d%02d%02d_%02d%02d%02d",
	       log_location,settings->filename_prefix,starting_time->tm_year+1900,
	       starting_time->tm_mon+1,starting_time->tm_mday,
	       starting_time->tm_hour,starting_time->tm_min,
	       starting_time->tm_sec);

      /* start a health checker process */
      if (health_control(HEALTH_START,failure)!=NO_ERROR)
	PrintStatus(failure);
      (void)snprintf(temp_status,BUFFSIZE,"recorder has PID = %d"
		     ,recorder_pid);
      PrintStatus(temp_status);
      status_server.recorder_pid=recorder_pid;
	
      status_server.is_recording=YES;
      BroadcastMessage("Recorder has started");

      /* write out the recording status file */
      if ((ret_res=WriteRecorderSettings(settings,failure))!=NO_ERROR)
	return(ret_res);

      time(&time_now);
      /* calculate the expected end time */
      end_time=start_time+timeinseconds(settings->record_time);

      status_server.recorder_start=start_time;
      status_server.recorder_end=end_time;

      /* reset warning flag */
      status_server.disk_warning_level=WARNING_NONE;

      /* free our memory */
      for (i=0;i<num_args+1;i++){
	free(recorder_args[i]);
      }
      free(recorder_args);

      /* copy the recorder structure to keep track of exactly the settings
         we've used here */
      copy_recorder(settings,
		    status_server.recording_settings);

      /* update the status so we can recover from server crashes */
      if ((upstatus=UpdateStatus(failure))!=NO_ERROR)
	return(upstatus);

      /* set the recording disk and path in settings */
      status_server.recording_to_disk=settings->recordingdisk;
      strcpy(status_server.recording_path,settings->directory_name);

      /* open the experiment filename report file */
      if (status_server.experiment_report!=NULL){
	/* close the already opened file */
	fclose(status_server.experiment_report);
	status_server.experiment_report=NULL;
      }
      if (strlen(experiment_report_filename)>0){
	snprintf(temp,BUFFSIZE,"Opening experiment report file %s\n",
		 experiment_report_filename);
	PrintLog(temp);
	if ((status_server.experiment_report=fopen(experiment_report_filename,"a"))==NULL){
	  PrintLog("unable to open experiment report file!\n");
	}
      }

      /* that's it, we're done */
      return(NO_ERROR);
    }
  } else if (command==RECSTOP){
    if (status_server.is_recording==NO){
      strcpy(failure,NOT_RECORDING_ERR);
      return(NOT_RECORDING);
    } else if (status_server.is_recording==YES){

      /* to stop the recorder we need to send it the unblockable kill signal SIGKILL */
      if ((kill(status_server.recorder_pid,SIGKILL))!=0){ /* something went wrong! */
	strcpy(failure,STOP_FAILED_ERR);
	return(STOP_FAILED);
      }

      /* the recorder has been stopped */
      (void)snprintf(buf,BUFFSIZE,"%s/recorder_%d_settings",tmp_directory,status_server.recorder_pid);
      /* we leave the recorder process number alone so UpdateStatus can clean */
      /* up the child for us and put the recorder output into the log */
      status_server.is_recording=NO;
      status_server.recorder_start=(time_t)0;

      /* do we need to stop a remote receiver? */
      if ((status_server.recording_settings->targetrecorder!=NULL)&&
	  (status_server.recording_settings->targetrecorder->evlbi_enabled==NO)){
	/* yes we do! */
	conjugate_host(status_server.recording_settings->targetrecorder->commonname,temp_host);
	snprintf(temp,BUFFSIZE,"<cmnd>receive-stop:%s</cmnd>",temp_host);
	if (PushRemote(temp,status_server.recording_settings->targetrecorder)!=NO_ERROR){
	  strcpy(failure,temp);
	}
      }

      /* change the directory now, since we're really trying to deprecate (since noone ever uses
         it anymore) disk cleaning and data checking */
      directory_default(failure);
      status_server.in_dir=NO;

      /* unset the recording disk and path */
      status_server.recording_to_disk=NULL;
      status_server.recording_path[0]='\0';

      /* call UpdateStatus */
      BroadcastMessage("Recorder has stopped");
      if ((upstatus=UpdateStatus(failure))!=NO_ERROR)
	return(upstatus);

      return(NO_ERROR);
      
    }      
  }
  return(NO_ERROR);
}

/*
  This routine handles the starting and stopping of the recorder
  health checker. We make this a separate routine so we can
  start the health checker by itself if it dies during the
  recording.
*/
int health_control(int command,char *failure){
  int health_pid,mypid,i;
  char **health_args=NULL,tmp[BUFFSIZE],outname[BUFFSIZE];
  recorder_health *free_health=NULL;
  remoterecorder *cyclerecorder=NULL;

  if (command==HEALTH_START){

    if (status_server.healthcheck_pid>0){
      /* health checker already running */
      strcpy(failure,HEALTH_RUNNING_ERR);
      return(HEALTH_RUNNING);
    }

    if (status_server.is_recording==NO){
      /* can't check the health of a non-existant recorder */
      strcpy(failure,HEALTH_NORECORDER_ERR);
      return(HEALTH_NORECORDER);
    }

    /* reset recorder health list */
    while(status_server.recstatus!=NULL){
      free_health=status_server.recstatus;
      status_server.recstatus=status_server.recstatus->next;
      free(free_health);
    }
    status_server.recstatus=NULL;
    
    /* start a health checker process */
    (void)snprintf(outname,BUFFSIZE,"%s/recorder_%d_output",
		   tmp_directory,status_server.recorder_pid);
    if ((health_pid=fork())==0){
      MALLOC(health_args,3*sizeof(char*));
/*       health_args=malloc(3*sizeof(char*)); */
      for (i=0;i<2;i++){
	MALLOC(health_args[i],BUFFSIZE*sizeof(char));
/* 	health_args[i]=malloc(BUFFSIZE*sizeof(char)); */
      }
      strcpy(health_args[0],health_checker_command);
      strcpy(health_args[1],outname);
      health_args[2]=NULL;
      /* get our pid */
      mypid=getpid();
      /* close the server socket */
      (void)close(connected_socket);
      (void)close(listen_socket);
      /* and close any other sockets we're listening to */
      for (cyclerecorder=status_server.remote_recorders;cyclerecorder!=NULL;
	   cyclerecorder=cyclerecorder->next){
	if (cyclerecorder->connectionsocket>0){
	  (void)close(cyclerecorder->connectionsocket);
	}
      }
      /* turn off the alarm */
      (void)alarm(0);
      (void)signal(SIGALRM,SIG_IGN);
      /* we need to change the process group id of the child */
      /* so a Ctrl-C to the parent doesn't kill the child */
      (void)setpgid(mypid,mypid); /* create a new process group */
      /* set the Ctrl-C handler back to default */
      (void)signal(SIGINT,SIG_DFL);
      /* replace the child with the health checker process */
      if ((execv(health_checker_command,health_args))==-1){ /* this is where the code ends for the child process */
	PrintStatus("didn't start health checker!\n");
	_exit(EXIT_FAILURE);
      }
    } else if (health_pid<0){ /* couldn't fork */
      strcpy(failure,HEALTH_FORK_FAIL_ERR);
      return(HEALTH_FORK_FAIL);
    }
    status_server.healthcheck_pid=health_pid;
    (void)snprintf(tmp,BUFFSIZE,"Started health checker with PID = %d",
		   health_pid);
    PrintStatus(tmp);
    return(NO_ERROR);

  } else if (command==HEALTH_STOP){

    if (status_server.healthcheck_pid==-1){
      strcpy(failure,HEALTH_NOTRUNNING_ERR);
      return(HEALTH_NOTRUNNING);
    }

    /* send the kill signal to the health checker process */
    kill(status_server.healthcheck_pid,SIGKILL);

    return(NO_ERROR);

  } else {
    strcpy(failure,HEALTH_BAD_CMND_ERR);
    return(HEALTH_BAD_CMND);
  }
  
}

/*
  This routine is called when either the client sends a data string,
  or when the defaults are loaded at the start of the server. It looks
  at the data request, and checks to make sure that the requested
  settings are valid, and will return an error if it is not.
*/
int data_handler(char *data,char *failure,recorder *settings){
  char variable[BUFFSIZE],value[BUFFSIZE],temp[BUFFSIZE],*u_p;
  char add_hostname[BUFFSIZE],add_commonname[BUFFSIZE],search_recorder[BUFFSIZE];
  char search_diskname[BUFFSIZE];
  char secret_ip[BUFFSIZE],rem_commonname[BUFFSIZE],rem_path[BUFFSIZE],rem_label[BUFFSIZE];
  int returncode=-1,secret_switch=0,secret_port;
  int year=-1,month=-1,date=-1,datecheck=-1,hour=-1,minute=-1,i_second=-1,timecheck=-1;
  int temp_time,disknumber=-1,tempdisk,datarate,hostfound=0,remargs,setvalue;
  int reset_switch=-1,add_recorderserver_port,add_data_communication_port,addargs;
  int rem_mounted,rem_maxrate,foundhost,add_evlbi_enabled,mod_udpenabled,mod_ipd;
  float add_tcpwindowsize;
  outputdisk *cycledisk=NULL,*newdisk=NULL,*cycledisk2=NULL;
  remoterecorder *newrecorder=NULL,*cyclerecorder=NULL,*cyclerecorder2=NULL;
  unsigned long long rem_freespace,rem_totalspace;
  struct hostent *addhost_host=NULL;

  /* does the data request match the format variable=data ? */
  if ((sscanf(data,"%[^=]=%s",variable,value))!=2){ /* no it doesn't */
    returncode=MALFORMED_DATA;
    strcpy(failure,MALFORMED_DATA_ERR);
    return(returncode);
  }
  /* recast the gather so we can keep spaces in the value */
  strcpy(value,strstr(data,"=")+1);

  /* which assignment are we changing? */
  /*   record_time       = the time to record data for */
  /*   record_start_date = the date to start the recording */
  /*   record_start_time = the time to start the recording */
  /*   directory_name    = the directory to put the files in */
  /*   filename_prefix   = the prefix for the output filenames */
  /*   compression       = the channels to record the data for */
  /*   vsib_mode         = the vsib mode for bit length */
  /*   bandwidth         = the recording bandwidth */
  /*   recordingdisk     = the disk to record to */
  /*   remrecorder       = specify only a remote recorder, useful for auto-disk selection */
  /*   fringetest        = whether this is a fringe-test */
  /*   experimentexecute = whether to execute experiments automatically */
  /*   diskaction        = what action to take when disk space becomes critical */
  /*   diskselection     = is the server allowed/required to choose the disk */
  /*   disk-add          = add a disk to the list of acceptable selections */
  /*   disk-remove       = remove a disk from the list of acceptable selections */
  /*   round_start       = whether to start on 10s boundaries or immediately */
  /*   vsib_device       = the location of the VSIB recording device */
  /*   filesize          = the size of each recorded file in blocks/seconds */
  /*   blocksize         = the size of each block in B */
  /*   clockrate         = the rate of the VSIB clock in MHz */
  /*   evlbi             = operate in eVLBI mode */
  /*   remote_host       = the hostname of the machine to connect to */
  /*   remote_port       = the port to connect to on the remote host */
  /*   tcp_window_size   = the size of the TCP window in kB */
  /*   mark5b            = whether to enable Mark5B operation */
  /*   udp               = whether to enable UDP operation and MTU size */
  /*   onebit            = whether to enable 1 bit recording */
  /*   add_host          = add a new remote recorder */
  /*   modify_host       = modify a remote recorder */
  /*   rem_host          = remove a remote recorder */
  /*   remote_disk       = update information about remote disks */
  /*   numbits           = (header) the number of bits used to encode each sample */
  /*   encoding          = (header) the way the data is encoded */
  /*   frequency         = (header) the frequencies of each channel */
  /*   polarisation      = (header) the polarisation of each channel */
  /*   sideband          = (header) the spectral inversion for each channel */
  /*   referenceant      = (header) the location of the reference antenna */
  /* some of these options can be set with specific data commands: */
  /*   s2mode            = sets compression */
  /* we can also reset back to defaults using */
  /*   reset=assignment */
  /* or reset everything using */
  /*   reset=all */
  /* note that you cannot reset the fringetest option - it must be */
  /* explicitly changed */

  if (strcmp(variable,"record_time")==0){

    /* this setting should be a number, ended by an h, m or an s */
    if (sscanf(value,"%d%s",&temp_time,temp)!=2){
      returncode=MALFORMED_RECTIME;
      strcpy(failure,MALFORMED_RECTIME_ERR);
      return(returncode);
    } else if ((strcmp(temp,"s")!=0)&&(strcmp(temp,"h")!=0)&&(strcmp(temp,"m")!=0)){ /* the time is not in seconds or hours */
      returncode=MALFORMED_RECTIME;
      strcpy(failure,MALFORMED_RECTIME_ERR);
      return(returncode);
    }

    strcpy(settings->record_time,value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"record_start_date")==0){

    /* this setting could be yyyy-mm-dd or yyyymmdd */
    /* the value will be stored as yyyymmdd */
    if (strlen(value)==10){
      if ((sscanf(value,"%4d-%2d-%2d",&year,&month,&date))!=3){ /* the date doesn't look like it should */
	returncode=MALFORMED_DATE;
	strcpy(failure,MALFORMED_DATE_ERR);
	return(returncode);
      }
    } else if (strlen(value)==8){
      if ((sscanf(value,"%4d%2d%2d",&year,&month,&date))!=3){ /* the date doesn't look like it should */
	returncode=MALFORMED_DATE;
	strcpy(failure,MALFORMED_DATE_ERR);
	return(returncode);
      }
    } else { /* the length of the string is wrong */
      returncode=MALFORMED_DATE;
      strcpy(failure,MALFORMED_DATE_ERR);
      return(returncode);
    }
    
    /* check the starting date to make sure it's valid and not in the past */
    if ((datecheck=check_date(year,month,date))==MONTH_RANGE){ /* the month is out of range */
      returncode=datecheck;
      strcpy(failure,MONTH_RANGE_ERR);
      return(returncode);
    } else if (datecheck==DATE_RANGE){ /* the date is not valid for the month, or at all */
      returncode=datecheck;
      strcpy(failure,DATE_RANGE_ERR);
      return(returncode);
    } else if (datecheck==IN_THE_PAST){ /* the starting date is in the past */
      returncode=datecheck;
      strcpy(failure,IN_THE_PAST_ERR);
      return(returncode);
    }

    /* finally, store the starting date */
    (void)snprintf(settings->record_start_date,BUFFSIZE,"%4d%02d%02d",year,month,date);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"record_start_time")==0){

    /* this setting could be hh:mm:ss or hhmmss */
    /* the value will be stored as hhmmss */
    fprintf(stdout,"the time length is %d\n",(int)strlen(value));
    if (strlen(value)==8){
      if ((sscanf(value,"%2d:%2d:%2d",&hour,&minute,&i_second))!=3){
	returncode=MALFORMED_TIME;
	strcpy(failure,MALFORMED_TIME_ERR);
	return(returncode);
      }
    } else if (strlen(value)==6){
      if ((sscanf(value,"%2d%2d%2d",&hour,&minute,&i_second))!=3){ /* the time doesn't look like it should */
	returncode=MALFORMED_TIME;
	strcpy(failure,MALFORMED_TIME_ERR);
	return(returncode);
      }
    } else { /* the length of the string is wrong */
      returncode=MALFORMED_TIME;
      strcpy(failure,MALFORMED_TIME_ERR);
      return(returncode);
    }

    /* check the starting time to make sure it's valid and not in the past */
    /* to do this, we need to know the date, so the date needs to be set first, otherwise give an error */
    if (strlen(settings->record_start_date)==0){
      returncode=DATE_NOT_SET;
      strcpy(failure,DATE_NOT_SET_ERR);
      return(returncode);
    }

    sscanf(settings->record_start_date,"%4d%2d%2d",&year,&month,&date);
    if ((timecheck=check_time(year,month,date,hour,minute,i_second))==HOUR_RANGE){ /* the hour is out of range */
      returncode=timecheck;
      strcpy(failure,HOUR_RANGE_ERR);
      return(returncode);
    } else if (timecheck==MINUTE_RANGE){ /* the minute is out of range */
      returncode=timecheck;
      strcpy(failure,MINUTE_RANGE_ERR);
      return(returncode);
    } else if (timecheck==SECOND_RANGE){ /* the second is out of range */
      returncode=timecheck;
      strcpy(failure,SECOND_RANGE_ERR);
      return(returncode);
    } else if (timecheck==IN_THE_PAST){ /* the start time is in the past */
      returncode=timecheck;
      strcpy(failure,IN_THE_PAST_ERR);
      return(returncode);
    }

    /* finally, store the starting time */
    (void)snprintf(settings->record_start_time,BUFFSIZE,"%02d%02d%02d",hour,minute,i_second);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"directory_name")==0){
    
    /* the directory name can be anything */
    /* if it is a fringe-test, then it is stored in the FringeCheck directory */
    if (settings->fringe_test==1){ /* it is a fringe-test */
      (void)snprintf(settings->directory_name,BUFFSIZE,"FringeCheck/");
    } else { 
      strcpy(settings->directory_name,value);
    }
    returncode=NO_ERROR;
      
  } else if (strcmp(variable,"filename_prefix")==0){

    /* the file prefix can be anything */
    strcpy(settings->filename_prefix,value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"compression")==0){

    /* the compression string usually four characters long and made from o and x */
    /*    o = discard this channel */
    /*    x = include this channel */
    /* there is one exception, the compression can be xo for some special modes */

    /* check that we're only getting o and x in the string */
    if (sscanf(value,"%[ox]",temp)!=1){ 
      /* Does not start with x or o */
      returncode=BAD_COMPRESSION;
      strcpy(failure,BAD_COMPRESSION_ERR);
      return(returncode);
    } else {
      if (strlen(temp)!=strlen(value)) { 
	/* Not all x and o */
	returncode=BAD_COMPRESSION;
	strcpy(failure,BAD_COMPRESSION_ERR);
	return(returncode);
      }
    }
    
    /* check the length */
    if (strlen(value)!=4){
      /* will work only if the string is xo or ox */
      if ((strcmp(value,"xo")!=0)&&
	  (strcmp(value,"ox")!=0)&&
	  (strcmp(value,"xxooooxx")!=0)&&
	  (strcmp(value,"ooxxxxoo")!=0)){
	returncode=BAD_COMPRESSION;
	strcpy(failure,BAD_COMPRESSION_ERR);
	return(returncode);
      }
    }

    /* we can store the compression settings */
    strcpy(settings->compression,value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"s2mode")==0){
    
    /* this data command can be used to quickly set the compression mode */
    /* based on what mode the S2 recorder is in */
    /*   an S2 mode of x = compression ooxx */
    /*   an S2 mode of a = compression oxox */
    if (strcmp(value,"x")==0){
      strcpy(settings->compression,"ooxx");
      returncode=NO_ERROR;
    } else if (strcmp(value,"a")==0){
      strcpy(settings->compression,"oxox");
      returncode=NO_ERROR;
    } else { /* setting not recognised */
      returncode=INVALID_S2_MODE;
      strcpy(failure,INVALID_S2_MODE_ERR);
      return(returncode);
    }
    
  } else if (strcmp(variable,"vsib_mode")==0){

    /* the vsib mode is either 3, for 8-bit data (default), */
    /* or 2, for 16-bit data */
    if ((atoi(value)!=2)&&(atoi(value)!=3)){ /* not a valid vsib mode */
      returncode=BAD_VSIB_MODE;
      strcpy(failure,BAD_VSIB_MODE_ERR);
      return(returncode);
    }

    /* otherwise, we store the vsib mode setting */
    settings->vsib_mode=atoi(value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"bandwidth")==0){
    
    /* the bandwidth in MHz must be a power of 2 */
    /* ie. 1, 2, 4, 8, 16, 32, 64 */
    if ((atoi(value)!=1)&&(atoi(value)!=2)&&(atoi(value)!=4)
	&&(atoi(value)!=8)&&(atoi(value)!=16)
	&&(atoi(value)!=32)&&(atoi(value)!=64)){ /* not a valid bandwidth */
      returncode=BAD_BANDWIDTH;
      strcpy(failure,BAD_BANDWIDTH_ERR);
      return(returncode);
    }

    /* otherwise, we store the bandwidth setting */
    settings->bandwidth=atoi(value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"recordingdisk")==0){

    if ((sscanf(value,"%d",&tempdisk))==1){
      disknumber=tempdisk; /* the disk was specified as a number */
      cycledisk=disk_one;
      while(cycledisk!=NULL){
	if (cycledisk->disknumber==disknumber){
	  break;
	}
	cycledisk=cycledisk->next;
      }
      if (cycledisk==NULL){
	/* haven't found the disk number in the local disks, search
	   the remote disks */
	for (cyclerecorder2=status_server.remote_recorders;
	     cyclerecorder2!=NULL;cyclerecorder2=cyclerecorder2->next){
	  for (cycledisk2=cyclerecorder2->available_disks;
	       cycledisk2!=NULL;cycledisk2=cycledisk2->next){
	    if (cycledisk2->disknumber==disknumber){
	      cycledisk=cycledisk2;
	      cyclerecorder=cyclerecorder2;
	      break;
	    }
	  }
	}
      }
    } else {
      /* search by name */
      if (strstr(value,":")!=NULL){
	/* must be a remote disk name, get the components */
	sscanf(value,"%[^:]:%s",rem_commonname,rem_path);
	/* now search for it */
	for (cyclerecorder2=status_server.remote_recorders;
	     cyclerecorder2!=NULL;cyclerecorder2=cyclerecorder2->next){
	  if (strcmp(cyclerecorder2->commonname,rem_commonname)==0){
	    /* check if it is an eVLBI recorder */
	    if (cyclerecorder2->evlbi_enabled==YES){
	      cycledisk=NULL;
	      cyclerecorder=cyclerecorder2;
	      break;
	    }
	    for (cycledisk2=cyclerecorder2->available_disks;
		 cycledisk2!=NULL;cycledisk2=cycledisk2->next){
	      if (strcmp(cycledisk2->diskpath,rem_path)==0){
		cycledisk=cycledisk2;
		cyclerecorder=cyclerecorder2;
		break;
	      }
	    }
	  }
	}
      } else {
	/* might be a local disk or a disk label from either a local
	   or remote disk */
	cycledisk=disk_one;
	while(cycledisk!=NULL){
	  if (strcmp(cycledisk->diskpath,value)==0){
	    break;
	  } else if (strcmp(cycledisk->disklabel,value)==0){
	    break;
	  }
	  cycledisk=cycledisk->next;
	}
	if (cycledisk==NULL){
	  for (cyclerecorder2=status_server.remote_recorders;
	       cyclerecorder2!=NULL;cyclerecorder2=cyclerecorder2->next){
	    for (cycledisk2=cyclerecorder2->available_disks;
		 cycledisk2!=NULL;cycledisk2=cycledisk2->next){
	      if (strcmp(cycledisk2->diskpath,value)==0){
		cycledisk=cycledisk2;
		cyclerecorder=cyclerecorder2;
		break;
	      } else if (strcmp(cycledisk2->disklabel,value)==0){
		cycledisk=cycledisk2;
		cyclerecorder=cyclerecorder2;
		break;
	      }
	    }
	  }
	}
      }
    }
	
    if ((cycledisk!=NULL)||(cyclerecorder!=NULL)){
      if (cycledisk!=NULL){
	datarate=recordingdatarate(settings);
	if (datarate>cycledisk->max_rate){
	  strcpy(failure,DATARATE_TOOFAST_ERR);
	  return(DATARATE_TOOFAST);
	}
      }
      settings->recordingdisk=cycledisk; /* remember this pointer */
      settings->targetrecorder=cyclerecorder;
      returncode=NO_ERROR;
    } else { /* we couldn't find the requested disk */
      returncode=DISKNOTFOUND;
      strcpy(failure,DISKNOTFOUND_ERR);
      /* note that we're leaving the recorder pointer as it was before this call */
    }
    
    return(returncode);

  } else if (strcmp(variable,"remrecorder")==0){

    /* have we been asked to go back to the local disks? */
    if (strcmp(value,"local")==0){
      /* yes we have, so we just select the first disk */
      settings->targetrecorder=NULL;
      settings->recordingdisk=disk_one;
    } else {
      cyclerecorder=NULL;
      /* has to be given as the remote recorder's common name */
      for (cyclerecorder2=status_server.remote_recorders;
	   cyclerecorder2!=NULL;cyclerecorder2=cyclerecorder2->next){
	if (strcmp(cyclerecorder2->commonname,value)==0){
	  /* we've found the right recorder, set the selected disk
	     to the first one (unless it is eVLBI) */
	  settings->targetrecorder=cyclerecorder2;
	  cyclerecorder=cyclerecorder2;
	  if (cyclerecorder2->evlbi_enabled==YES){
	    settings->recordingdisk=NULL;
	  } else {
	    settings->recordingdisk=cyclerecorder2->available_disks;
	  }
	  break;
	}	
      }
      
      /* did we find the specified recorder? */
      if (cyclerecorder==NULL){
	/* no we didn't! */
	strcpy(failure,REMREC_UNKNOWN_ERR);
	return(REMREC_UNKNOWN);
      }
    }

    return(NO_ERROR);

  } else if (strcmp(variable,"fringetest")==0){

    /* the value should be yes or no */
    if (strcmp(value,"yes")==0){ /* is a fringe-test */
      settings->fringe_test=YES;
      /* set the directory name to be FringeCheck/ */
      (void)snprintf(settings->directory_name,BUFFSIZE,"FringeCheck/");
    } else if (strcmp(value,"no")==0){ /* is not a fringe-test */
      settings->fringe_test=NO;
      /* if the directory name was already specified, it may need to be changed */
      if (settings->directory_name[0]!='\0'){ /* directory name has been set */
	if (strcmp(settings->directory_name,"FringeCheck/")==0){
	  /* directory is set to be the FringeCheck directory */
	  /* reset the directory name */
	  data_handler("reset=directory_name",temp,settings);
	} /* otherwise the directory is already not in the FringeCheck directory */
      }
    } else { /* invalid value */
      strcpy(failure,FRINGE_TEST_VAL_ERR);
      return(FRINGE_TEST_VAL);
    }
    
    return(NO_ERROR);

  } else if (strcmp(variable,"experimentexecute")==0){

    /* the value should be on or off */
    if (strcmp(value,"on")==0){ /* auto experiment control is on */
      status_server.execute_experiment=AUTO_EXPERIMENT_YES;
    } else if (strcmp(value,"off")==0){ /* auto experiment control is off */
      status_server.execute_experiment=AUTO_EXPERIMENT_NO;
    } else { /* invalid value */
      strcpy(failure,AUTOCONTROL_VAL_ERR);
      return(AUTOCONTROL_VAL);
    }

    return(NO_ERROR);

  } else if (strcmp(variable,"diskaction")==0){

    /* the value can be: none to do nothing
                         stop to stop recording
			 swap to swap disks */
    if (strcmp(value,"none")==0){
      status_server.low_disk_action=CRITICAL_DONOTHING;
    } else if (strcmp(value,"stop")==0){
      status_server.low_disk_action=CRITICAL_STOP;
    } else if (strcmp(value,"swap")==0){
      status_server.low_disk_action=CRITICAL_SWITCH;
    } else { /* invalid value */
      strcpy(failure,INVALIDDISKACTION_ERR);
      return(INVALIDDISKACTION);
    }

    return(NO_ERROR);

  } else if (strcmp(variable,"diskselection")==0){

    /* the value can be: on_any    to let the server choose any disk
                         on_local  to let the server choose only a local disk
			 on_remote to let the server choose only a remote disk
			 on_list   to let the server choose only from the user list
			 off       to disable this function */
    if (strcmp(value,"on_any")==0){
      settings->auto_disk_select=AUTODISK_ANY;
    } else if (strcmp(value,"on_local")==0){
      settings->auto_disk_select=AUTODISK_LOCAL;
    } else if (strcmp(value,"on_remote")==0){
      settings->auto_disk_select=AUTODISK_REMOTE;
    } else if (strcmp(value,"on_list")==0){
      settings->auto_disk_select=AUTODISK_LIST;
    } else if (strcmp(value,"off")==0){
      settings->auto_disk_select=AUTODISK_DISABLED;
    } else {
      /* option not recognised */
      strcpy(failure,DISKSEL_UNKNOWN_ERR);
      return(DISKSEL_UNKNOWN);
    }

    return(NO_ERROR);

  } else if ((strcmp(variable,"disk-add")==0)||
	     (strcmp(variable,"disk-remove")==0)){

    /* what will be our value? */
    if (strcmp(variable,"disk-add")==0){
      setvalue=YES;
    } else if (strcmp(variable,"disk-remove")==0){
      setvalue=NO;
    }

    /* can only accept one argument, but this can use wildcards */
    search_recorder[0]='\0';
    search_diskname[0]='\0';
    /* what format is the argument in? */
    if (strstr(value,":")!=NULL){
      /* there is a recorder specification */
      sscanf(value,"%[^:]:%[^:]",search_recorder,search_diskname);
    } else {
      /* we should assume local disks */
      strcpy(search_recorder,"local");
      strcpy(search_diskname,value);
    }
    if ((strcmp(search_recorder,"local")==0)||
	(strcmp(search_recorder,"*")==0)){
      /* go through local disks first */
      for (cycledisk=disk_one;cycledisk!=NULL;
	   cycledisk=cycledisk->next){
	if (cycledisk->is_mounted==NO){
	  continue;
	}
	if ((strcmp(search_diskname,cycledisk->diskpath)==0)||
	    (strcmp(search_diskname,cycledisk->disklabel)==0)||
	    (strcmp(search_diskname,"*")==0)){
	  cycledisk->is_acceptable=setvalue;
	}
      }
    }

    if (strcmp(search_recorder,"local")!=0){
      /* now go through the remote disks */
      for (cyclerecorder=status_server.remote_recorders;
	   cyclerecorder!=NULL;cyclerecorder=cyclerecorder->next){
	if ((strcmp(search_recorder,cyclerecorder->commonname)==0)||
	    (strcmp(search_recorder,"*")==0)){
	  for (cycledisk=cyclerecorder->available_disks;
	       cycledisk!=NULL;cycledisk=cycledisk->next){
	    if (cycledisk->is_mounted==NO){
	      continue;
	    }
	    if ((strcmp(search_diskname,cycledisk->diskpath)==0)||
		(strcmp(search_diskname,cycledisk->disklabel)==0)||
		(strcmp(search_diskname,"*")==0)){
	      cycledisk->is_acceptable=setvalue;
	    }
	  }
	}
      }
    }

    return(NO_ERROR);

  } else if (strcmp(variable,"round_start")==0){
    
    /* should be "on" or "off" */
    if ((strcmp(value,"on"))==0){
      /* start recording only on 10s boundaries */
      status_server.rounded_start=ROUNDSTART_YES;
    } else if ((strcmp(value,"off"))==0){
      /* start recording immediately when called */
      status_server.rounded_start=ROUNDSTART_NO;
    } else {
      strcpy(failure,ROUNDED_INVALID_ERR);
      return(ROUNDED_INVALID);
    }

    return(NO_ERROR);

  } else if (strcmp(variable,"vsib_device")==0){

    /* the location of the VSIB recording device - can really
       be anything, so we won't do any checks here */
    strcpy(settings->vsib_device,value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"filesize")==0){

    /* the size of each recorded file in blocks (default)
       or seconds */
    if (value[strlen(value)-1]=='s'){
      /* given in seconds */
      value[strlen(value)-1]='\0';
      settings->filesize_is_time=YES;
    } else
      settings->filesize_is_time=NO;

    settings->filesize_or_time=atof(value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"blocksize")==0){

    /* the size of each block in B */
    settings->blocksize=atof(value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"clockrate")==0){

    /* the rate of the VSIB clock in MHz */
    settings->clockrate=atof(value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"mark5b")==0){

    /* whether to use Mark5B recording 
       to enable, send on
       to disable, send off */
    if (strcmp(value,"on")==0){
      settings->mark5b_enabled=YES;
    } else if (strcmp(value,"off")==0){
      settings->mark5b_enabled=NO;
    } else {
      strcpy(failure,MARK5B_INVALID_ERR);
      return(MARK5B_INVALID);
    }
    returncode=NO_ERROR;

  } else if (strcmp(variable,"udp")==0){

    /* the udp MTU to use */
    if ((atoi(value)>=0)&&(atoi(value)<65536))
      settings->udp_enabled=atoi(value);
    else {
      strcpy(failure,UDP_INVALID_ERR);
      return(UDP_INVALID);
    }
    returncode=NO_ERROR;

  } else if (strcmp(variable,"onebit")==0){

    /* whether to enable 1 bit recording */
    if (strcmp(value,"on")==0){
      settings->onebit_enabled=YES;
    } else if (strcmp(value,"off")==0){
      settings->onebit_enabled=NO;
    } else {
      strcpy(failure,ONEBIT_INVALID_ERR);
      return(ONEBIT_INVALID);
    }
    returncode=NO_ERROR;

  } else if (strcmp(variable,"add_host")==0){
    /* this variable allows you to set a remote recorder that will be
       then "attached" to this recorder_server */

    /* the format of this variable should be:
       commonname,hostname,data_communication_port,tcpwindowsize,evlbimode[,recorder_server_port] 
       where commonname: is a nickname for the host, eg. cavsi1 
             hostname  : the fully qualified hostname, eg. cavsi1.atnf.csiro.au 
             data_communication_port : the port to send data to
	     tcpwindowsize : the TCP window size for data transfer
	     evlbimode : whether the recorder is for eVLBI (1) or just remote storage (0)
             recorder_server_port : an optional value, use only if the remote recorder
	                            server is listening on a port other than 50080 
       there is also some ultra-secret options, that should be used by the recorder_server
       experts: 
       - if one extra option is given with the value of -1 (so the recorder_server_port 
       must be specified in this case), then recorder_server will not attempt to contact
       the remote host with its description.
       - if one extra option is given with the value of -2, then recorder_server will not
       attempt to contact the remote host with its description, and it will ignore the
       IP address (or hostname) that was supplied, and will instead substitute the currently
       connected IP for it.
       - if three extra options are given, the first should be any value other than -1,
       and the second and third should be the IP address and port that recorder_server
       should report to the remote host. These options should be used to specify how to
       communicate between servers when the "add_host" request comes from a different
       network than the one that should be used for cross-server communication.
    */
    /* try to parse the string */
    addargs=sscanf(value,"%[^,],%[^,],%d,%f,%d,%d,%d,%[^,],%d",add_commonname,add_hostname,
		   &add_data_communication_port,&add_tcpwindowsize,&add_evlbi_enabled,
		   &add_recorderserver_port,&secret_switch,secret_ip,&secret_port);
    if (addargs<5){
      /* not enough arguments present */
      strcpy(failure,HOST_SPECBAD_ERR);
      return(HOST_SPECBAD);
    } else {
      if (addargs==5){
	/* use default recorder_server port */
	add_recorderserver_port=LISTENPORT;
      }
      /* got all our information, make a new host, but first check that
         the host doesn't already exist in our list */
      hostfound=0;
      for (cyclerecorder=status_server.remote_recorders;
	   cyclerecorder!=NULL;cyclerecorder=cyclerecorder->next){
	if ((strcmp(cyclerecorder->commonname,add_commonname)==0)||
	    ((strcmp(cyclerecorder->hostname,add_hostname)==0)&&
	     (cyclerecorder->data_communication_port==add_data_communication_port)&&
	     (cyclerecorder->recorderserver_port==add_recorderserver_port)&&
	     (((cyclerecorder->evlbi_enabled==YES)&&(add_evlbi_enabled==1))||
	      ((cyclerecorder->evlbi_enabled==NO)&&(add_evlbi_enabled==0))))){
	  hostfound=1;
	  break;
	}
      }
      if (hostfound==0){
	/* the host doesn't already exist, so we create it */
	MALLOC(newrecorder,sizeof(remoterecorder));
	strcpy(newrecorder->commonname,add_commonname);
	if ((addargs==7)&&(secret_switch==-2)){
	  /* use the connected IP for this instead */
	  strcpy(newrecorder->hostname,connected_ip);
	} else {
	  strcpy(newrecorder->hostname,add_hostname);
	}
	newrecorder->recorderserver_port=add_recorderserver_port;
	if (add_evlbi_enabled==1){
	  newrecorder->evlbi_enabled=YES;
	} else {
	  newrecorder->evlbi_enabled=NO;
	}
	newrecorder->data_communication_port=add_data_communication_port;
	newrecorder->tcp_window_size=add_tcpwindowsize;
	/* try to lookup the host we just got told about */
/* 	newrecorder->host=gethostbyname(newrecorder->hostname); */
	addhost_host=gethostbyname(newrecorder->hostname);
/* 	if (newrecorder->host==NULL){ */
	if (addhost_host==NULL){
	  /* couldn't get an IP for this host, so we have to fail */
	  strcpy(failure,HOST_LOOKUPFAIL_ERR);
	  FREE(newrecorder);
	  return(HOST_LOOKUPFAIL);
	}
	MALLOC(newrecorder->host,sizeof(struct hostent));
	copy_hostent(newrecorder->host,addhost_host);
	newrecorder->udp_enabled=NO;
	newrecorder->ipd=0;
	newrecorder->receiver=NULL;
	newrecorder->available_disks=NULL;
	if ((secret_switch==-1)||(secret_switch==-2)){
	  newrecorder->connectionsocket=connected_socket;
	  snprintf(temp,BUFFSIZE,"accepting, using socket %d for remote host communication",
		   newrecorder->connectionsocket);
 	  PrintStatus(temp);
	  connected_socket=-1;
	}
	/* and add it to the list */
	newrecorder->next=status_server.remote_recorders;
	status_server.remote_recorders=newrecorder;
	if (newrecorder->evlbi_enabled==NO){
	  if (((addargs==7)&&((secret_switch!=-1)&&(secret_switch!=-2)))||
	      (addargs!=7)){
	    /* send our details to the remote recorder */
	    conjugate_host(newrecorder->commonname,add_commonname);
	    if (addargs==9){
	      snprintf(temp,BUFFSIZE,"<data>add_host=%s,%s,%d,%f,%d,%d,-1</data>",
		       add_commonname,secret_ip,
		       newrecorder->data_communication_port,
		       newrecorder->tcp_window_size,newrecorder->evlbi_enabled,
		       secret_port);
	    } else {
	      snprintf(temp,BUFFSIZE,"<data>add_host=%s,%s,%d,%f,%d,%d,-2</data>",
		       add_commonname,connected_on_ip,
		       newrecorder->data_communication_port,
		       newrecorder->tcp_window_size,newrecorder->evlbi_enabled,
		       LISTENPORT);
	    }
	    AddRemote(temp,newrecorder);
	  }
	  status_server.update_remote_disks=YES;
	}
      }
      
    }
    returncode=NO_ERROR;

  } else if (strcmp(variable,"modify_host")==0){
    /* this variable allows you to change the settings of a remote recorder
       that can't be set using add_host */

    /* the format of this variable should be: 
       commonname,udpenabled,ipd
       where commonname: is the nickname for the host, eg. cavsi1 
             udpenabled: is 1 to enable UDP transfers for this host, 0 to disable 
             ipd: the interpacket delay for UDP transfers */
    /* try to parse the string */
    addargs=sscanf(value,"%[^,],%d,%d",add_commonname,&mod_udpenabled,
		   &mod_ipd);
    if (addargs!=3){
      /* not enough arguments present */
      strcpy(failure,HOST_MODIFYBAD_ERR);
      return(HOST_MODIFYBAD);
    }
    /* find the host */
    for (cyclerecorder=status_server.remote_recorders;
	 cyclerecorder!=NULL;cyclerecorder=cyclerecorder->next){
      if (strcmp(cyclerecorder->commonname,add_commonname)==0){
	if (mod_udpenabled>0){
	  cyclerecorder->udp_enabled=mod_udpenabled;
	} else {
	  cyclerecorder->udp_enabled=NO;
	}
	cyclerecorder->ipd=mod_ipd;
	break;
      }
    }
    returncode=NO_ERROR;

  } else if (strcmp(variable,"rem_host")==0){
    /* this variable allows you to remove a remote recorder from the
       list of disks */
    
    /* the format of this variable should be: 
       commonname 
       where commonname: is the nickname the host is known by, eg. cavsi1 */
    /* try to parse the string */
    addargs=sscanf(value,"%s",add_commonname);
    if (addargs!=1){
      /* wrong number of arguments */
      strcpy(failure,HOST_REMSPECBAD_ERR);
      return(HOST_REMSPECBAD);
    }
    /* find the host */
    hostfound=0;
    for (cyclerecorder=status_server.remote_recorders;
	 cyclerecorder!=NULL;cyclerecorder=cyclerecorder->next){
      if (strcmp(cyclerecorder->commonname,add_commonname)==0){
	/* remove this host */
	hostfound=1;
	if ((cyclerecorder->receiver!=NULL)||
	    (settings->targetrecorder==cyclerecorder)||
	    ((status_server.recording_settings->targetrecorder==cyclerecorder)&&
	     (status_server.is_recording==YES))){
	  /* this recorder is being used, so we can't remove it */
	  strcpy(failure,HOST_REMUSED_ERR);
	  return(HOST_REMUSED);
	}
	RemoveRemote(cyclerecorder,REMOTEHOST_REMOVE_IMMEDIATE);
	break;
      }
    }
    if (hostfound==0){
      strcpy(failure,REMHOST_UNKNOWN_ERR);
      return(REMHOST_UNKNOWN);
    }
    returncode=NO_ERROR;

  } else if (strcmp(variable,"remote_disk")==0){
    /* this variable should only be used internally for communication between 
       servers that have been linked together by the add_host variable */
    
    /* the format of this variable should be:
       commonname,path,freespace[,totalspace,label,maxrate,mounted] 
       where commonname: is that the recorder is known by to this server
                   path: is the path to the disk
              freespace: the amount of free space on the disk 
             totalspace: the total capacity of the disk 
                  label: the label associated with the disk
                maxrate: the maximum recording rate supported by the disk
                mounted: is the disk mounted (YES/NO) 
       if the disk state hasn't changed, the server should communicate only
       the freespace of each disk */
    /* try to parse the string */
    remargs=sscanf(value,"%[^,],%[^,],%llu,%llu,%[^,],%d,%d",rem_commonname,
		   rem_path,&rem_freespace,&rem_totalspace,rem_label,&rem_maxrate,
		   &rem_mounted);
    /* replace question marks with spaces in the disk label */
    while(strstr(rem_label,"?")!=NULL){
      (strstr(rem_label,"?"))[0]=' ';
    }
    snprintf(temp,BUFFSIZE,"args = %d",remargs);
/*     PrintStatus(temp); */
    snprintf(temp,BUFFSIZE,"%s %s %llu %llu %s %d %d",rem_commonname,rem_path,
	     rem_freespace,rem_totalspace,rem_label,rem_maxrate,rem_mounted);
/*     PrintStatus(temp); */
    if (remargs<3){
      /* not enough arguments present */
      strcpy(failure,DISK_SPECBAD_ERR);
      return(DISK_SPECBAD);
    }
    /* check that we know about this remote host */
    foundhost=0;
    for (cyclerecorder=status_server.remote_recorders;cyclerecorder!=NULL;
	 cyclerecorder=cyclerecorder->next){
      if (strcmp(cyclerecorder->commonname,rem_commonname)==0){
	foundhost=1;
	break;
      }
    }
    if (foundhost==0){
      /* don't know about this host, return a failure */
      strcpy(failure,UNKNOWN_REMHOST_ERR);
      return(UNKNOWN_REMHOST);
    }
    if (remargs==3){
/*       PrintStatus("getting free space remote disk update"); */
      /* just a free space update */
      for (cyclerecorder=status_server.remote_recorders;cyclerecorder!=NULL;
	   cyclerecorder=cyclerecorder->next){
	if (strcmp(cyclerecorder->commonname,rem_commonname)==0){
	  for (cycledisk=cyclerecorder->available_disks;cycledisk!=NULL;
	       cycledisk=cycledisk->next){
	    if (strcmp(cycledisk->diskpath,rem_path)==0){
	      /* found the disk we're updating */
	      cycledisk->freespace=rem_freespace;
	      /* we're finished */
	      return(NO_ERROR);
	    }
	  }
	}
      }
    } else if (remargs==7){
/*       PrintStatus("getting full remote disk update"); */
      /* a full disk update */
      for (cyclerecorder=status_server.remote_recorders;cyclerecorder!=NULL;
	   cyclerecorder=cyclerecorder->next){
	if (strcmp(cyclerecorder->commonname,rem_commonname)==0){
	  for (cycledisk=cyclerecorder->available_disks;cycledisk!=NULL;
	       cycledisk=cycledisk->next){
	    if (strcmp(cycledisk->diskpath,rem_path)==0){
	      /* found the disk we're updating */
	      cycledisk->freespace=rem_freespace;
	      cycledisk->totalspace=rem_totalspace;
	      strcpy(cycledisk->disklabel,rem_label);
	      cycledisk->max_rate=rem_maxrate;
	      cycledisk->is_mounted=rem_mounted;
	      /* we're finished */
	      return(NO_ERROR);
	    }
	  }
/* 	  PrintStatus("adding new remote disk"); */
	  /* if we get here, we haven't found the disk, so we must add
	     it to this recorder */
	  MALLOC(newdisk,sizeof(outputdisk));
	  strcpy(newdisk->diskpath,rem_path);
	  if (strcmp(rem_label,"none")==0){
	    newdisk->disklabel[0]='\0';
	  } else {
	    strcpy(newdisk->disklabel,rem_label);
	  }
	  next_disknumber++;
	  newdisk->disknumber=next_disknumber;
	  newdisk->max_rate=rem_maxrate;
	  newdisk->is_mounted=rem_mounted;
	  newdisk->get_label=NO;
	  newdisk->is_acceptable=NO;
	  newdisk->filesystem[0]='\0';
	  newdisk->freespace=rem_freespace;
	  newdisk->totalspace=rem_totalspace;
	  /* add it to the list */
	  if (cyclerecorder->available_disks!=NULL){
	    cyclerecorder->available_disks->previous=newdisk;
	  }
	  newdisk->next=cyclerecorder->available_disks;
	  newdisk->previous=NULL;
	  cyclerecorder->available_disks=newdisk;

	  break;
	}
      }
    } else {
      /* wrong number of arguments present */
      strcpy(failure,DISK_SPECBAD_ERR);
      return(DISK_SPECBAD);
    }

    returncode=NO_ERROR;

  } else if (strcmp(variable,"numbits")==0){
    /* this variable will be passed directly to the vsib_record */
    /* command for entry in the output file header */
    
    /* the number of bits used to encode each sample, typically */
    /* 2, 8 or 10 */
    settings->numbits=atoi(value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"ipd")==0){
    /* this variable will be passed directly to the vsib_record */
    /* command for entry in the output file header */
    
    /* The delay in usec between UDP packets */
    settings->ipd=atoi(value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"encoding")==0){
    /* this variable will be passed directly to the vsib_record */
    /* command for entry in the output file header */

    /* either "AT" or "VLBA" depending on whether the (2-bit) data */
    /* is encoded as offset-binary or sign-magnitude */
    strcpy(settings->encoding,value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"frequency")==0){
    /* this variable will be passed directly to the vsib_record */
    /* command for entry in the output file header */

    /* the lower band edge frequency of each channel in MHz, as a */
    /* whitespace delimited list of floating point numbers. Note */
    /* that this is the lowest sky frequency of the band. For inverted */
    /* spectra this equals the effective LO minus the bandwidth. */
    
    /* note that since this parameter is whitespace delimited, it needs */
    /* to be passed to this routine as underscore delimited. we need */
    /* to replace the underscores with spaces */
    while((u_p=strstr(value,"_"))!=NULL)
      *u_p=' ';
    strcpy(settings->frequency,value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"polarisation")==0){
    /* this variable will be passed directly to the vsib_record */
    /* command for entry in the output file header */
    
    /* the polarisation of each channel, as a whitespace delimited */
    /* list of R or L (for Rcp and Lcp). the number of values must */
    /* equal the number of channels */
    
    /* note that since this parameter is whitespace delimited, it needs */
    /* to be passed to this routine as underscore delimited. we need */
    /* to replace the underscores with spaces */
    while((u_p=strstr(value,"_"))!=NULL)
      *u_p=' ';
    strcpy(settings->polarisation,value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"sideband")==0){
    /* this variable will be passed directly to the vsib_record */
    /* command for entry in the output file header */

    /* the sideband of each channel, as a whitespace delimited list of */
    /* U or L (for non-inverted or inverted spectra). note the upper or */
    /* lower sideband corresponds to the net sideband after all IF down */
    /* conversion and any inversion of the samplers */

    /* note that since this parameter is whitespace delimited, it needs */
    /* to be passed to this routine as underscore delimited. we need */
    /* to replace the underscores with spaces */
    while((u_p=strstr(value,"_"))!=NULL)
      *u_p=' ';
    strcpy(settings->sideband,value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"referenceant")==0){
    /* this variable will be passed directly to the vsib_record */
    /* command for entry in the output file header */

    /* the actual antenna position used for an "antenna" where there might */
    /* be some ambiguity. currently this is only useful for the ATCA where */
    /* the reference antenna could be any of the many pad positions. For the */
    /* ATCA the "W" form of the pad positions should be used */
    strcpy(settings->referenceant,value);
    returncode=NO_ERROR;

  } else if (strcmp(variable,"reset")==0){

    returncode=NO_ERROR;
    if (strcmp(value,"record_time")==0){
      reset_switch=RESET_RECORD_TIME;
    } else if (strcmp(value,"record_start_date")==0){
      reset_switch=RESET_RECORD_START_DATE;
    } else if (strcmp(value,"record_start_time")==0){
      reset_switch=RESET_RECORD_START_TIME;
    } else if (strcmp(value,"directory_name")==0){
      reset_switch=RESET_DIRECTORY_NAME;
    } else if (strcmp(value,"filename_prefix")==0){
      reset_switch=RESET_FILENAME_PREFIX;
    } else if (strcmp(value,"compression")==0){
      reset_switch=RESET_COMPRESSION;
    } else if (strcmp(value,"vsib_mode")==0){
      reset_switch=RESET_VSIB_MODE;
    } else if (strcmp(value,"bandwidth")==0){
      reset_switch=RESET_BANDWIDTH;
    } else if (strcmp(value,"recordingdisk")==0){
      reset_switch=RESET_DISKS;
    } else if (strcmp(value,"fringetest")==0){
      returncode=FRINGETEST_RESET;
      strcpy(failure,FRINGETEST_RESET_ERR);
    } else if (strcmp(value,"vsib_device")==0){
      reset_switch=RESET_DEVICE;
    } else if (strcmp(value,"filesize")==0){
      reset_switch=RESET_FILESIZE;
    } else if (strcmp(value,"blocksize")==0){
      reset_switch=RESET_BLOCKSIZE;
    } else if (strcmp(value,"clockrate")==0){
      reset_switch=RESET_CLOCKRATE;
    } else if (strcmp(value,"mark5b")==0){
      reset_switch=RESET_MARK5B;
    } else if (strcmp(value,"udp")==0){
      reset_switch=RESET_UDP;
    } else if (strcmp(value,"diskselection")==0){
      reset_switch=RESET_AUTODISK;
    } else if (strcmp(value,"all")==0){
      reset_switch=RESET_ALL;
    } else { /* the value is not a recognised data name */
      returncode=INVALID_RESET;
      strcpy(failure,INVALID_RESET_ERR);
    }
    if ((LoadSettings(settings,reset_switch,failure))!=NO_ERROR){
      returncode=RESET_FAILED;
      strcpy(failure,RESET_FAILED_ERR);
    }
    return(returncode);

  } else {
    
    /* this means that the requested data doesn't exist */
    returncode=BAD_DATA;
    strcpy(failure,BAD_DATA_ERR);
    return(returncode);

  }
    

  /* if we've made it this far, returncode will be NO_ERROR */
  return(returncode);

}

/*
  This routine checks that the time passed to it is an actual time, and not in the past.
  It assumes that the date portion of the time has already been checked, and is valid and
  not in the past.
*/
int check_time(int year,int month,int date,int hour,int minute,int second){
  time_t time_now,timecheck;

  /* get the current time */
  time(&time_now);
  assign_time(year,month,date,hour,minute,second,&timecheck);
    
  /* check for valid hour */
  if ((hour<0)||(hour>23))
    return(HOUR_RANGE);

  /* check for valid minute */
  if ((minute<0)||(minute>59))
    return(MINUTE_RANGE);

  /* check for valid second */
  if ((second<0)||(second>59))
    return(SECOND_RANGE);

  /* check that the time isn't in the past */
  if (timecheck<time_now)
    return(IN_THE_PAST);

  /* everything checks out */
  return(NO_ERROR);
}
		 
/*
  This routine checks that the date passed to it is an actual date, and not in the past
*/
int check_date(int year,int month,int date){
  time_t time_now;
  time_t datecheck;

  /* get the current date */
  time(&time_now);

  /* check for valid month */
  if ((month<1)||(month>12))
    return(MONTH_RANGE);

  /* check for valid day-of-month */
  if ((date<1)||(date>31))
    return(DATE_RANGE);

  switch (month){
  case 4: case 6: case 9: case 11:
    if (date>30)
      return(DATE_RANGE);
    break;
  case 2:
    /* determine if it's a leap year */
    if (xmody(year,4)!=0){
      if (date>28)
	return (DATE_RANGE);
    } else if (xmody(year,400)==0){
      if (date>29)
	return (DATE_RANGE);
    } else if (xmody(year,100)==0){
      if (date>28)
	return (DATE_RANGE);
    } else {
      if (date>28)
	return (DATE_RANGE);
    }
    break;
  }

  /* check that the date isn't in the past */
  /* we do this by checking that 23:59:59 on the date given is */
  /* in the future */
  assign_time(year,month,date,23,59,59,&datecheck);
  if (time_now>=datecheck)
    return(IN_THE_PAST);

  return(NO_ERROR);
}

/*
  This routine takes the modulo of x/y
*/
int xmody(int x,int y){
  while(x>=y)
    x-=y;
  return(x);
}

/*
  Send the confirmation message back to the client, or the error
  code and error message if the request has failed.
*/
void send_confirmation(int errorcode,char *errormessage,int sock){
  char tag[BUFFSIZE];
  int taglen=0;

  if (errorcode==NO_ERROR){ /* successful completion of request */
    strcpy(tag,"<succ />");
    taglen=(int)strlen(tag);
  } else if (errorcode==STATUS_COMPLETE){ /* status messages */
    (void)snprintf(tag,BUFFSIZE,"<status>%s</status>",errormessage);
    taglen=(int)strlen(tag);
  } else { /* something went wrong */
    (void)snprintf(tag,BUFFSIZE,"<fail>error %d: %s</fail>",errorcode,errormessage);
    taglen=(int)strlen(tag);
    PrintStatus(tag);
  }
  if (send(sock, tag, taglen+1, 0) != taglen+1)
    PrintStatus("Failed to send confirmation to client.");
}

/*
  This routine is called if a message is sent to the server.
  The message is printed to the screen, unless it is a zero
  length message, which causes an error.
*/
int message_handler(char *message,char *failure){
  int returncode=-1;

  failure[0]='\0';
  if (strlen(message)<1){
    returncode=SHORT_MESG;
    strcpy(failure,SHORT_MESG_ERR);
    return(returncode);
  }
   
  PrintStatus("Message received from client:");
  PrintStatus(message);

  return(NO_ERROR);
}

/*
  The main routine creates the socket, binds to it for listening,
  and waits until a connection is made, before calling the
  HandleClient routine.
*/
int main(int argc, char *argv[]) {
  int doy,i,timeremaining,activity;
  struct sockaddr_in mesgserver, mesgclient,caddr;
  socklen_t length_ptr;
  char failmsg[BUFFSIZE],logname[BUFFSIZE],logmesg[BUFFSIZE],tmp[BUFFSIZE],tmp2[BUFFSIZE];
  time_t time_now;
  struct tm *temp_time;
  recorder *settings;
  sigset_t block_alarm;
  fd_set readfds;
  remoterecorder *cyclerecorder=NULL;
  unsigned int clientlen;

  length_ptr=sizeof(struct sockaddr_in);

  if (argc>1){
    for (i=1;i<argc;i++){
      if ((strcmp(argv[i],"-v")==0)||
	  (strcmp(argv[i],"--version")==0)){
	printf("recorder_server, Copyright (c) 2006  Jamie Stevens\n");
	printf("recorder_server comes with ABSOLUTELY NO WARRANTY and is\n");
	printf("licensed under the GPL.\n");
	printf("This is free software, and you are welcome to redistribute\n");
	printf("it under certain conditions.\n");
	strcpy(tmp,source_version);
	strncpy(tmp2,tmp+25,4);
	printf("source version: %s\n",tmp2);
	strcpy(tmp,header_version);
	strncpy(tmp2,tmp+25,4);
	printf("header version: %s\n",tmp2);
      } else {
	printf("only the command line arguments -v/--version are supported\n");
      }
    }
    exit(0);
  }

  /* open the log file */
  time(&time_now);
  status_server.server_start=time_now; /* the startup time of the server */
  temp_time=gmtime(&time_now);
  doy=temp_time->tm_yday+1; /* the current day of the year, in GMT */
  (void)snprintf(logname,BUFFSIZE,"%s/recorder_log_%4d%03d",log_location,(temp_time->tm_year)+1900,doy);
  if ((logfile=fopen(logname,"a"))==NULL) {
    sprintf(failmsg, "Cannot open logfile \"%s\" for writing!\n", logname);
    Die(failmsg, listen_socket);
  }
  logday=doy;

  /* startup messages */
  (void)snprintf(logmesg,BUFFSIZE,"Recorder Server started at %02d:%02d:%02d %02d/%02d/%4d",
	  temp_time->tm_hour,temp_time->tm_min,temp_time->tm_sec,temp_time->tm_mday,(temp_time->tm_mon)+1,(temp_time->tm_year)+1900);
  PrintStatus(logmesg);

  /* allocate memory for server settings */
  MALLOC(settings,sizeof(recorder));
/*   settings=malloc(sizeof(recorder)); */

  /* initialise the server status */
  initialise_status(settings);

  /* initialise the recorder settings */
  if ((GetSettings(settings,RESET_INITIAL,failmsg))!=NO_ERROR)
    Die(failmsg,listen_socket); /* die if an error occurs on startup */

  /* catch any kill signals so we can clean up properly */
  (void)signal(SIGINT,server_stop);

  /* ignore the SIGHUP signal so we won't quit if our terminal is taken */
  /* away from us */
  (void)signal(SIGHUP,SIG_IGN);
  /* also ignore the SIGPIPE signal, so we don't die if our clients stop 
     and close before we do something with them */
  (void)signal(SIGPIPE,SIG_IGN);

  /* call UpdateStatus every STATUS_UPDATE seconds */
  (void)signal(SIGALRM,UpdateStatusSignal);
  (void)alarm(STATUS_UPDATE);
  /* we need to be able to block this signal while user commands are 
     executed, otherwise we might be interrupted */
  sigemptyset(&block_alarm);
/*   sigaddset(&block_alarm,SIGALRM); /\* set up the signal blocker *\/ */
  sigaddset(&block_alarm,SIGCHLD); /* don't want any interruptions */
  sigaddset(&block_alarm,SIGINT);  /* none at all */

  /* Create the TCP socket */
  if ((listen_socket = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
    Die("Failed to create socket",listen_socket);
  PrintLog("Server socket created.");
  socket_options(listen_socket,SOCKETOPTIONS_MAIN);

  /* Construct the server sockaddr_in structure */
  memset(&mesgserver, 0, sizeof(mesgserver));       /* Clear struct */
  mesgserver.sin_family = AF_INET;                  /* Internet/IP */
  mesgserver.sin_addr.s_addr = htonl(INADDR_ANY);   /* Incoming addr */
  mesgserver.sin_port = htons(LISTENPORT);          /* server port */
  
  /* Bind the server socket */
  if (bind(listen_socket, (struct sockaddr *) &mesgserver,
	   sizeof(mesgserver)) < 0)
    Die("Failed to bind the server socket",listen_socket);

  /* Listen on the server socket */
  if (listen(listen_socket, MAXPENDING) < 0)
    Die("Failed to listen on server socket",listen_socket);
  (void)snprintf(logmesg,BUFFSIZE,"Server listening on port %d",LISTENPORT);
  PrintStatus(logmesg);
           
  /* Run until cancelled */
  while (1) {
    /* clear the list of sockets to listen on */
    FD_ZERO(&readfds);
    /* now go through and figure out which sockets we should be listening to */
    if (listen_socket!=fileno(logfile)){
      FD_SET(listen_socket,&readfds); /* the main socket of course */
    } else {
      snprintf(logmesg,BUFFSIZE,"listen socket has become the logfile fd!");
      PrintStatus(logmesg);
    }
    /* check for any remote recorder sockets */
    for (cyclerecorder=status_server.remote_recorders;cyclerecorder!=NULL;
	 cyclerecorder=cyclerecorder->next){
      if ((cyclerecorder->connectionsocket>0)&&
          (cyclerecorder->connectionsocket!=fileno(logfile))){
	FD_SET(cyclerecorder->connectionsocket,&readfds);
	cyclerecorder->comms_wait=NO;
      }
    }
    
    /* wait for a connection */
    activity=select(MAXPENDING+3,&readfds,NULL,NULL,NULL);

    clientlen = sizeof(mesgclient);
    if (activity<0){
      continue;
    }
    if (FD_ISSET(listen_socket,&readfds)){
      /* we have a new client connection */
      if ((connected_socket=accept(listen_socket,(struct sockaddr *)&mesgclient,
				   &clientlen))<0){
	snprintf(logmesg,BUFFSIZE,"Could not accept connection from IP %s",
		 inet_ntoa(mesgclient.sin_addr));
	PrintStatus(logmesg);
	close(connected_socket);
	continue;
      } else if (connected_socket==fileno(logfile)){
	snprintf(logmesg,BUFFSIZE,"accept has accepted socket connection on logfile fd!");
	PrintStatus(logmesg);
	continue;
      }
      socket_options(connected_socket,SOCKETOPTIONS_MAIN);
      (void)snprintf(logmesg,BUFFSIZE,"Client connected from IP %s",inet_ntoa(mesgclient.sin_addr));
      PrintStatus(logmesg);
      strcpy(connected_ip,inet_ntoa(mesgclient.sin_addr));
      getsockname(connected_socket,(struct sockaddr *)&caddr,&length_ptr);
      strcpy(connected_on_ip,inet_ntoa(caddr.sin_addr));
      is_remote_host=NO;
    } else {
      /* maybe it was one of our remote recorders */
      for (cyclerecorder=status_server.remote_recorders;cyclerecorder!=NULL;
	   cyclerecorder=cyclerecorder->next){
	if (FD_ISSET(cyclerecorder->connectionsocket,&readfds)){
	  connected_socket=cyclerecorder->connectionsocket;
	  snprintf(logmesg,BUFFSIZE,"Getting message from remote recorder %s",
		   cyclerecorder->commonname);
	  PrintStatus(logmesg);
	  is_remote_host=YES;
	  break; /* no use looking for more connections */
	}
      }
    }

    /* block UpdateStatus signals */
    sigprocmask(SIG_BLOCK,&block_alarm,NULL);

    /* we set an alarm to go to UpdateStatus in 5 seconds if we haven't
       returned here, to keep the server from blocking indefinitely */
    hanging=YES;
    timeremaining=alarm(5);
    HandleClient(connected_socket,settings);
    /* OK we're back, so we reset the alarm to the time that was remaining
       before we went away (we assume a correct response would have taken
       negligible time) */
    hanging=NO;
    (void)alarm(timeremaining);

    /* settings may have changed, so we need to update our settings pointer */
    settings=status_server.current_settings;

    /* unblock UpdateStatus signals */
    sigprocmask(SIG_UNBLOCK,&block_alarm,NULL);
  }
}

/*
  This routine is called when the server is shut down nicely by the user.
*/
void server_stop(int sig){
  char time_string[BUFFSIZE],buf[BUFFSIZE],fail[BUFFSIZE];
  experiment *free_experiment=NULL;
  registered_clients *free_clients=NULL;
  outputdisk *free_disk=NULL;
  recorder_health *free_health=NULL;
  rectarget *free_targets;

  /* update status so recovery is easier */
  (void)UpdateStatus(fail);

  /* tell everybody we're stopping */
  IssueWarning("The recorder server has been stopped");
  thetimenow(time_string);
  (void)snprintf(buf,BUFFSIZE,"Recorder server stopped at %s",time_string);
  PrintStatus(buf);

  /* free memory allocations */
  /* experiment list */
  while(next_experiment!=NULL){
    free_experiment=next_experiment;
    next_experiment=next_experiment->next;
    free(free_experiment->record_settings);
    free(free_experiment);
  }
  /* message clients */
  while(message_clients!=NULL){
    free_clients=message_clients;
    message_clients=message_clients->next;
    free(free_clients);
  }
  /* recording disks */
  while (local_disks!=NULL){
    free_disk=local_disks;
    local_disks=local_disks->next;
    free(free_disk);
  }
  while (remote_disks!=NULL){
    free_disk=remote_disks;
    remote_disks=remote_disks->next;
    free(free_disk);
  }
  /* recorder health list */
  while(status_server.recstatus!=NULL){
    free_health=status_server.recstatus;
    status_server.recstatus=status_server.recstatus->next;
    free(free_health);
  }
  /* recorder target list */
  while (status_server.target_list!=NULL){
    free_targets=status_server.target_list;
    status_server.target_list=status_server.target_list->next;
    free(free_targets->recorder_settings);
    free(free_targets);
  }

  /* last bits of house-keeping */
  fflush(NULL); /* write everything to disk/screen that is pending */
  fclose(logfile);
  close(listen_socket); /* close the network socket */

  /* we're out of here */
  exit(EXIT_SUCCESS);

}

/*
  This routine is called by routines when they want to reset the defaults
  for particular settings.
*/
int LoadSettings(recorder *settings,int RESET,char *failmsg){
  FILE *defaults=NULL;
  int line,i_response;
  char in_line[BUFFSIZE],variable[BUFFSIZE],value[BUFFSIZE],data[BUFFSIZE],response[BUFFSIZE];
  char temp[BUFFSIZE];

  /* defaults */
  PrintStatus("Resetting to default settings.");
  if (RESET_RECORD_TIME & RESET)
    strcpy(settings->record_time,"12h");    /* default of 12 hours; this parameter is required */
  if (RESET_RECORD_START_DATE & RESET)
    settings->record_start_date[0] = '\0';  /* no default, but not necessarily needed */
  if (RESET_RECORD_START_TIME & RESET)
    settings->record_start_time[0] = '\0';  /* no default, but not necessarily needed */
  if (RESET_DIRECTORY_NAME & RESET)
    settings->directory_name[0]    = '\0';  /* no default, and is required */
  if (RESET_FILENAME_PREFIX & RESET)
    settings->filename_prefix[0]   = '\0';  /* no default, and is required */
  if (RESET_COMPRESSION & RESET)
    strcpy(settings->compression,"xxxx");   /* all channels enabled by default */
  if (RESET_VSIB_MODE & RESET)
    settings->vsib_mode            = 3;     /* 8-bit recording default */
  if (RESET_BANDWIDTH & RESET)
    settings->bandwidth            = 16;    /* 16 MHz bandwidth default */
  if (RESET_DISKS & RESET)
    settings->recordingdisk        = NULL;  /* no default recording disk */
  if (RESET_DEVICE & RESET)
    strcpy(settings->vsib_device,
	   DEFAULT_DEVICE);                 /* default VSIB recorder location is /dev/vsib */
  if (RESET_FILESIZE & RESET){
    settings->filesize_or_time     = DEFAULT_FILESIZE; /* default size of file is 10 seconds */
    settings->filesize_is_time     = DEFAULT_FILETIME;
  }
  if (RESET_BLOCKSIZE & RESET)
    settings->blocksize            = DEFAULT_BLOCKSIZE; /* default size of each block is 32 kB */
  if (RESET_CLOCKRATE & RESET)
    settings->clockrate            = DEFAULT_CLOCKRATE; /* default clockrate for VSIB is 32 MHz */
  if (RESET_MARK5B & RESET)
    settings->mark5b_enabled       = DEFAULT_MARK5B; /* do not use Mark5B by default */
  if (RESET_UDP & RESET)
    settings->udp_enabled          = DEFAULT_UDP;   /* do not use udp by default */
  if (RESET_NUMBITS & RESET)
    settings->numbits              = DEFAULT_NUMBITS; /* no default header information */
  if (RESET_IPD & RESET)
    settings->ipd                  = DEFAULT_IPD; /* No IPD by default */
  if (RESET_AUTODISK & RESET)
    settings->auto_disk_select     = AUTODISK_DISABLED; /* disallow automatic disk selection */
  if (RESET_ENCODING & RESET)
    strcpy(settings->encoding,DEFAULT_ENCODING);
  if (RESET_FREQUENCY & RESET)
    strcpy(settings->frequency,DEFAULT_FREQUENCY);
  if (RESET_POLARISATION & RESET)
    strcpy(settings->polarisation,DEFAULT_POLARISATION);
  if (RESET_SIDEBAND & RESET)
    strcpy(settings->sideband,DEFAULT_SIDEBAND);
  if (RESET_REFERENCEANT)
    strcpy(settings->referenceant,DEFAULT_REFERENCEANT);
  if (RESET==RESET_INITIAL){
    settings->fringe_test          = NO;    /* set default status to be non-fringe-test */
  }

  /* are there user/system defaults? */
  if ((defaults=fopen("/etc/recorder_server.conf","r"))!=NULL){ /* file exists */
    if (RESET==RESET_INITIAL)
      PrintStatus("Reading system defaults...");
    line=0;
    while((fgets(in_line,BUFFSIZE,defaults))!=NULL){
      line++;
      /* the lines of this file should look like */
      /* variable   value */
      /* only the data variables are allowed to be specified */
      /* comment lines are allowed by placing a # symbol at the start of the line */
      if ((sscanf(in_line,"%s %s",variable,value))==2){ /* format is correct */
	if (in_line[0]!='#'){ /* not a comment line */
	  /* check whether we've been asked to set this variable */
	  if (((strcmp(variable,"record_time")==0)&&(RESET_RECORD_TIME & RESET))||
	      ((strcmp(variable,"record_start_date")==0)&&(RESET_RECORD_START_DATE & RESET))||
	      ((strcmp(variable,"record_start_time")==0)&&(RESET_RECORD_START_TIME & RESET))||
	      ((strcmp(variable,"directory_name")==0)&&(RESET_DIRECTORY_NAME & RESET))||
	      ((strcmp(variable,"filename_prefix")==0)&&(RESET_FILENAME_PREFIX & RESET))||
	      ((strcmp(variable,"compression")==0)&&(RESET_COMPRESSION & RESET))||
	      ((strcmp(variable,"vsib_mode")==0)&&(RESET_VSIB_MODE & RESET))||
	      ((strcmp(variable,"bandwidth")==0)&&(RESET_BANDWIDTH & RESET))||
	      ((strcmp(variable,"recordingdisk")==0)&&(RESET_DISKS & RESET))||
	      ((strcmp(variable,"vsib_device")==0)&&(RESET_DEVICE & RESET))||
	      ((strcmp(variable,"filesize")==0)&&(RESET_FILESIZE & RESET))||
	      ((strcmp(variable,"blocksize")==0)&&(RESET_BLOCKSIZE & RESET))||
	      ((strcmp(variable,"clockrate")==0)&&(RESET_CLOCKRATE & RESET))||
	      ((strcmp(variable,"mark5b")==0)&&(RESET_MARK5B & RESET))||
	      ((strcmp(variable,"udp")==0)&&(RESET_UDP & RESET))||
	      ((strcmp(variable,"numbits")==0)&&(RESET_NUMBITS & RESET))||
	      ((strcmp(variable,"ipd")==0)&&(RESET_IPD & RESET))||
	      ((strcmp(variable,"encoding")==0)&&(RESET_ENCODING & RESET))||
	      ((strcmp(variable,"frequency")==0)&&(RESET_FREQUENCY & RESET))||
	      ((strcmp(variable,"polarisation")==0)&&(RESET_POLARISATION & RESET))||
	      ((strcmp(variable,"sideband")==0)&&(RESET_SIDEBAND & RESET))||
	      ((strcmp(variable,"referenceant")==0)&&(RESET_REFERENCEANT & RESET))||
	      ((strcmp(variable,"fringetest")==0)&&(RESET_INITIAL == RESET))){
	    /* we'll use the data_handler routine to check the values are legal */
	    (void)snprintf(data,BUFFSIZE,"%s=%s",variable,value);
	    if ((i_response=data_handler(data,response,settings))!=NO_ERROR){
	      (void)snprintf(temp,BUFFSIZE,"error on line %d of recorder_server.conf!",line);
	      PrintStatus(temp);
	      PrintStatus(in_line);
	      (void)snprintf(temp,BUFFSIZE,"error %d: %s",i_response,response);
	      PrintStatus(temp);
	    }
	  }
	}
      } else if (in_line[0]!='#'){ /* not a comment line */
	(void)snprintf(temp,BUFFSIZE,"error on line %d of recorder_server.conf!",line);
	PrintStatus(temp);
	(void)snprintf(temp,BUFFSIZE,"%s is incorrectly specified",in_line);
	PrintStatus(temp);
      }
    }
    fclose(defaults);
  }
  return(NO_ERROR);
}

/*
  This routine determines whether the diskpath passed to it represents
  a mounted disk.
*/
int check_disk_mount(char *diskpath,char *filesystem){
  FILE *proc_mounts=NULL;
  char fline[BUFFSIZE],*curr_delim,part[BUFFSIZE];
  int found_mounted=NO,nel,y=0;

  if ((proc_mounts=fopen("/proc/mounts","r"))==NULL){
    return(NO);
  }

  while((fgets(fline,BUFFSIZE,proc_mounts))!=NULL){
    nel=0;
    curr_delim=fline;
    while((tokstr(&curr_delim," ",part))!=NULL){
      if (nel==1){
	/* second element in the line is the mount point */
	if (strcmp(part,diskpath)==0){
	  found_mounted=YES;
	  y=1;
	} else {
	  y=0;
	}
      } else if (nel==2){
	/* third element in the line is the filesystem */
	if (y==1){
	  strcpy(filesystem,part);
	}
      }
      nel++;
    }
  }

  fclose(proc_mounts);

  return(found_mounted);
}

/*
  This routine is called at the start of main to set the defaults for
  the parameters in the recorder structure. It is also used to detect
  whether there are user defined defaults (in /etc/recorder_server.conf).
  If there has been a server crash while the recorder server was running,
  the file /tmp/recorder_settings will exist, and this routine will
  take over control of the recording, using the settings from this file.
*/
int GetSettings(recorder *settings,int RESET,char *failmsg){
  FILE *recorders=NULL;
  char in_line[BUFFSIZE],temp[BUFFSIZE],temp_label[BUFFSIZE];
  char temp_path[BUFFSIZE],temp_station[BUFFSIZE],temp_stname[BUFFSIZE];
  int lerr=-1,i,dir_check,max_rate,got_station=0;
  outputdisk *newDisk=NULL,*cycledisk=NULL;
  struct stat check_dir;

  PrintStatus("Getting the server settings.");

  /* initialise the default settings */
  if ((lerr=LoadSettings(settings,RESET,failmsg))!=NO_ERROR)
    return(lerr);

  /* get the list of available recording devices */
  if ((recorders=fopen(disks_config_file,"r"))==NULL){ /* oops, no available disks! */
    strcpy(failmsg,NO_DISKS_ERR);
    return(NO_DISKS);
  }
  /* this file should have a number of disk paths in it, one per line */
  /* comment lines are allowed, but only if the first character on the line is a # */
  PrintStatus("Finding recording targets.");
  i=0;
  next_disknumber=0;
  while((fgets(in_line,BUFFSIZE,recorders))!=NULL){
    if (in_line[0]!='#'){ /* make sure the line is not a comment line */
      if (in_line[strlen(in_line)-1]=='\n')
	in_line[strlen(in_line)-1]='\0'; /* remove the newline character */
      if ((sscanf(in_line,"%s = %s",temp_station,temp_stname))==2){
	if (strcmp(temp_station,"STATION")==0){
	  strcpy(STATION_NAME,temp_stname);
	  got_station=1;
	}
      }
      if ((sscanf(in_line,"%s %d",temp_path,&max_rate))==2){
	/* first check that what has been specified is a directory */
	if ((dir_check=stat(temp_path,&check_dir))==0){ /* the name exists */
	  if (S_ISDIR(check_dir.st_mode)){ /* and it's a directory */
	    MALLOC(newDisk,sizeof(outputdisk));
	    strcpy(newDisk->diskpath,temp_path);
	    newDisk->max_rate=max_rate;
	    i++; /* first disk is number 1 */
	    next_disknumber++;
	    newDisk->disknumber=next_disknumber;
	    newDisk->is_mounted=newDisk->get_label=
	      check_disk_mount(newDisk->diskpath,newDisk->filesystem);
	    newDisk->is_acceptable=NO;
	    if (newDisk->get_label==YES){
	      strcpy(temp_label,temp_path);
	      fprintf(stderr,"searching for label for %s\n",temp_label);
	      get_drive_serial(temp_label,newDisk->filesystem,failmsg,DRIVES_LABEL);
	      fprintf(stderr,"found label = %s\n",temp_label);
	      strcpy(newDisk->disklabel,temp_label);
	      newDisk->get_label=NO;
	    }
	    newDisk->next=NULL; /* always adding at the end of the list */
	    if (settings->recordingdisk==NULL){ /* this is the first disk */
	      newDisk->previous=NULL;
	      settings->recordingdisk=newDisk; /* by default, the recording disk is the first one */
	    } else {
	      cycledisk=settings->recordingdisk;
	      while(cycledisk->next!=NULL)
		cycledisk=cycledisk->next;
	      newDisk->previous=cycledisk;
	      cycledisk->next=newDisk;
	    }
	    newDisk=NULL;
	  }
	}
      }
    }
  }
  fclose(recorders);
  /* check we got a station name */
  if (got_station==0){
    /* we don't know where we are */
    strcpy(failmsg,NO_STATION_NAME_ERR);
    return(NO_STATION_NAME);
  }
  /* set the global disk list */
  local_disks=settings->recordingdisk;
  disk_one=local_disks;
  (void)snprintf(temp,BUFFSIZE,"Found %d recording devices.",i);
  PrintStatus(temp);
  settings->n_recorders=i; /* how many recording disks were read */
		 
  /* if the server exited, there should be a file detailing what it was doing */
  /* we'll load it to recover */
/*   (void)snprintf(temp,BUFFSIZE,"%s/recorder_server_status",tmp_directory); */
/*   if ((defaults=fopen(temp,"r"))!=NULL){ */
/*     fclose(defaults); */
/*     if ((lerr=LoadServerSettings(settings,failmsg))!=NO_ERROR){ */
/*       return(lerr); */
/*     } */
/*   } /\* otherwise we start afresh *\/ */
  
  return(NO_ERROR);

}

int LoadServerSettings(recorder *settings,char *failure){
  FILE *serversettings=NULL;
  char buf[BUFFSIZE],tmp[BUFFSIZE];
  int i,tmp1,tmp2,tmp4,lres,em=0;
  long int tmp3;

  PrintStatus("Loading the server recovery settings.");
  (void)snprintf(tmp,BUFFSIZE,"%s/recorder_server_status",tmp_directory);
  if ((serversettings=fopen(tmp,"r"))==NULL){
    strcpy(failure,STATUS_READ_FAIL_ERR);
    return(STATUS_READ_FAIL);
  }
  i=0;
  while((fgets(buf,BUFFSIZE,serversettings))!=NULL){
    buf[strlen(buf)-1]='\0'; /* get rid of \n character */
    i++;
    switch (i){
    case 1: /* recording status */
      if ((sscanf(buf,"%d %d %ld %d",&tmp1,&tmp2,&tmp3,&tmp4))!=4){
	strcpy(failure,STATUS_CORRUPT_ERR);
	return(STATUS_CORRUPT);
      }
      if (tmp1==YES){
	/* check vsib_record is still running */
	if (checkrunning(tmp2,vsib_record_command)){ /* it is */
	  status_server.is_recording=tmp1;
	  status_server.recorder_pid=tmp2;
	  status_server.recorder_start=(time_t)tmp3;
	} else { /* it's not */
	  status_server.is_recording=NO;
	  status_server.recorder_pid=-1;
	  status_server.recorder_start=(time_t)0;
	}
      }
      if (tmp4>0){
	/* check health checker is still running */
	if (checkrunning(tmp4,health_checker_command)){ /* it is */
	  status_server.healthcheck_pid=tmp4;
	} else {
	  status_server.healthcheck_pid=-1;
	}
      }
      break;
    case 2: /* experiment mode */
      if ((sscanf(buf,"%d %d",&tmp1,&tmp2))!=2){
	strcpy(failure,STATUS_CORRUPT_ERR);
	return(STATUS_CORRUPT);
      }
      if (tmp1==EXPERIMENT_MANUAL){ /* single record mode */
	/* just load the manual settings */
	if (status_server.is_recording==YES){
	  if ((lres=LoadRecorderSettings(settings,failure))!=NO_ERROR){
	    return(lres);
	  }
	  /* set the recorder's end time */
	  status_server.recorder_end=status_server.recorder_start
	    +(time_t)timeinseconds(status_server.current_settings->record_time);
	}
      } else if (tmp1==EXPERIMENT_QUEUE){ /* experiment mode */
	em=1;
      }
      status_server.execute_experiment=tmp2; /* auto experiment execution */
      break;
    default: /* load the experiment profiles */
      if ((lres=LoadExperiment(buf,failure,1))!=NO_ERROR){
	return(lres);
      }
    }
  }
  fclose(serversettings);
  return(NO_ERROR);
}

/*
  This routine reads an experiment profile from disk.
*/
int LoadExperiment(char *profilename,char *failure,int recovermode){
  FILE *profile=NULL;
  char root_path[BUFFSIZE],full_path[BUFFSIZE],buf[BUFFSIZE],tmp[BUFFSIZE];
  char tag[BUFFSIZE],value[BUFFSIZE],val2[BUFFSIZE];
  experiment *newExperiment=NULL;
  char experiment_id[BUFFSIZE],time_start[BUFFSIZE],time_end[BUFFSIZE],fringe_check[BUFFSIZE];
  char channels[BUFFSIZE],required_disk[BUFFSIZE],directory_name[BUFFSIZE],filename[BUFFSIZE];
  char encoding[BUFFSIZE],frequency[BUFFSIZE],polarisation[BUFFSIZE],sideband[BUFFSIZE];
  char referenceant[BUFFSIZE],*e_p,*s_p,clean_time[BUFFSIZE],round_start[BUFFSIZE];
  char filesize[BUFFSIZE],vsib_device[BUFFSIZE];
  char mark5b[BUFFSIZE],required_recorder[BUFFSIZE],*station_loc=NULL;
  int vsibmode,bandwidth,sthour,stminute,stsecond,stday,stmonth,styear;
  int enhour,enminute,ensecond,enday,enmonth,enyear,lres,imload=0,numbits, ipd;
  int blocksize,clockrate,udp;
  time_t time_now,start_time,end_time;
  struct tm *temp_time;
  recorder *newSettings=NULL,*runSettings=NULL;

  /* the path where the profiles are stored */
  strcpy(root_path,experiment_location);
  
  (void)snprintf(full_path,BUFFSIZE,"%s/%s",root_path,profilename);
  if ((profile=fopen(full_path,"r"))==NULL){
    strcpy(failure,PROFILE_NOT_FOUND_ERR);
    /* if we generate an error during startup, then the server will die */
    /* there are no "terminal" errors in this section, so if we're in */
    /* recovermode, we'll just return success - the experiment of course */
    /* won't be loaded though */
    if (recovermode==1) return (NO_ERROR);
    else return(PROFILE_NOT_FOUND);
  }

  /* an experiment profile looks like */
  /* necessary tags: */
  /* EXPERIMENT_ID = (name of experiment) */
  experiment_id[0]='\0';
  /* TIME_START = (start time as HH:MM:SS dd/mm/yyyy in UTC) */
  time_start[0]='\0';
  /* TIME_END   = (end time as HH:MM:SS dd/mm/yyyy in UTC) */
  time_end[0]='\0';
  /* FRINGE_CHECK = (yes or no to whether this is a fringe-check experiment) */
  fringe_check[0]='\0';
  /* CHANNELS = (four character string of x and o) */
  channels[0]='\0';
  /* VSIBMODE = (the vsib mode) */
  vsibmode=0;
  /* BANDWIDTH = (the recording bandwidth in MHz) */
  bandwidth=0;
  /* optional tags: */
  /* REQUIRED_DISK = (the name of the disk to record to, or its number) [defaults to best disk] */
  required_disk[0]='\0';
  /* REQUIRED_RECORDER = (the name of the remote recorder to record to) [defaults to local] */
  required_recorder[0]='\0';
  /* DIRECTORY_NAME = (the name of the directory to save data to) [defaults to the experiment name] */
  directory_name[0]='\0';
  /* FILENAME = (the prefix of the files to create) [defaults to the experiment name and station] */
  filename[0]='\0';
  /* CLEAN_TIME = (the length of time to keep the data for before deleting) [defaults to infinity] */
  clean_time[0]='\0';
  /* ROUND_START = (yes or no to restrict record start to a 10s boundary) [defaults to yes] */
  round_start[0]='\0';
  /* VSIB_DEVICE = (the location of the VSIB recorder) [no default] */
  vsib_device[0]='\0';
  /* FILESIZE = (the length of the file in seconds or blocks) [defaults to 10s] */
  filesize[0]='\0';
  /* BLOCKSIZE = (the size of each block in B) [defaults to 32000] */
  blocksize=-1;
  /* CLOCKRATE = (the rate of the VSIB clock in MHz) [defaults to 32] */
  clockrate=-1;
  /* MARK5B = (use Mark5B recorder) [defaults to off] */
  mark5b[0]='\0';
  /* UDP = (MTU size) [default to off] */
  udp=0;
  /* header tags */
  /* NUMBITS = (the number bits used to encode each sample) [no default] */
  numbits=-1;
    /* IPD - delay between UDP packets */
  ipd = DEFAULT_IPD;
  /* ENCODING = (the data encoding method) [defaults to AT] */
  encoding[0]='\0';
  /* FREQUENCY = (the lower band edge frequencies in MHz) [no default] */
  frequency[0]='\0';
  /* POLARISATION = (the polarisation of each channel) [no default] */
  polarisation[0]='\0';
  /* SIDEBAND = (the inversion of each channel) [no default] */
  sideband[0]='\0';
  /* REFERENCEANT = (the location of the reference antenna) [no default] */
  referenceant[0]='\0';

  while ((fgets(buf,BUFFSIZE,profile))!=NULL){
    if ((strlen(buf)==0)||(buf[0]=='\n')){
      /* skip empty lines */
      continue;
    }
    if (buf[0]=='#'){
      /* skip comment lines */
      continue;
    }
    if ((sscanf(buf,"%s = %s",tag,value))!=2){
      strcpy(failure,PROFILE_CORRUPT_ERR);
      if (recovermode==1) return (NO_ERROR);
      else return(PROFILE_CORRUPT);
    }
    if (strcmp(tag,"EXPERIMENT_ID")==0){
      /* the experiment id must be the same as the filename */
      if (strcmp(value,profilename)!=0){
	/* not the same */
	strcpy(failure,PROFILE_MISMATCH_ERR);
	return(PROFILE_MISMATCH);
      }
      strcpy(experiment_id,value);
    } else if (strcmp(tag,"TIME_START")==0){
      if (strcmp(value,"immediate")==0){
	/* asked for immediate loading */
	/* only works if we're not in experiment mode */
	if (status_server.experiment_mode==EXPERIMENT_QUEUE){
	  strcpy(failure,NO_IMMEDIATE_ERR);
	  if (recovermode==1) return(NO_ERROR);
	  else return(NO_IMMEDIATE);
	}
	imload=1;
      } else {
	/* need to rescan the buf string */
	if ((sscanf(buf,"%s = %s %s",tag,value,val2))!=3){
	  strcpy(failure,PROFILE_CORRUPT_ERR);
	  if (recovermode==1) return (NO_ERROR);
	  else return(PROFILE_CORRUPT);
	}
	(void)snprintf(time_start,BUFFSIZE,"%s %s",value,val2);
      }
    } else if (strcmp(tag,"TIME_END")==0){
      if (strcmp(value,"immediate")==0){
	/* asked for immediate loading */
	if (status_server.experiment_mode==EXPERIMENT_QUEUE){
	  strcpy(failure,NO_IMMEDIATE_ERR);
	  if (recovermode==1) return(NO_ERROR);
	  else return(NO_IMMEDIATE);
	}
	imload=1;
      } else {
	/* need to rescan the buf string */
	if ((sscanf(buf,"%s = %s %s",tag,value,val2))!=3){
	  strcpy(failure,PROFILE_CORRUPT_ERR);
	  if (recovermode==1) return (NO_ERROR);
	  else return(PROFILE_CORRUPT);
	}
	(void)snprintf(time_end,BUFFSIZE,"%s %s",value,val2);
      }
    } else if (strcmp(tag,"FRINGE_CHECK")==0){
      strcpy(fringe_check,value);
    } else if (strcmp(tag,"CHANNELS")==0){
      strcpy(channels,value);
    } else if (strcmp(tag,"VSIBMODE")==0){
      vsibmode=atoi(value);
    } else if (strcmp(tag,"BANDWIDTH")==0){
      bandwidth=atoi(value);
    } else if (strcmp(tag,"REQUIRED_DISK")==0){
      strcpy(required_disk,value);
    } else if (strcmp(tag,"REQUIRED_RECORDER")==0){
      strcpy(required_recorder,value);
    } else if (strcmp(tag,"DIRECTORY_NAME")==0){
      strcpy(directory_name,value);
    } else if (strcmp(tag,"FILENAME")==0){
      /* check if the user has included the %S string, which
         should be substituted for the station name */
      station_loc=strstr(value,"%S");
      if (station_loc==NULL){
	/* not specified */
	strcpy(filename,value);
      } else {
	/* replace the S with s (so it becomes a string
	   replacement printf format) */
	*(station_loc+1)='s';
	/* do the substitution */
	(void)snprintf(filename,BUFFSIZE,value,STATION_NAME);
      }
    } else if (strcmp(tag,"ROUND_START")==0){
      strcpy(round_start,value);
    } else if (strcmp(tag,"VSIB_DEVICE")==0){
      strcpy(vsib_device,value);
    } else if (strcmp(tag,"FILESIZE")==0){
      strcpy(filesize,value);
    } else if (strcmp(tag,"BLOCKSIZE")==0){
      blocksize=atoi(value);
    } else if (strcmp(tag,"CLOCKRATE")==0){
      clockrate=atoi(value);
    } else if (strcmp(tag,"MARK5B")==0){
      strcpy(mark5b,value);
    } else if (strcmp(tag,"NUMBITS")==0){
      numbits=atoi(value);
    } else if (strcmp(tag,"IPD")==0){
      ipd=atoi(value);
    } else if (strcmp(tag,"ENCODING")==0){
      strcpy(encoding,value);
    } else if (strcmp(tag,"FREQUENCY")==0){
      /* values are space delimited, so get all values */
      e_p=strstr(buf,"="); /* find the equals sign */
      while((*e_p==' ')||(*e_p=='='))
	e_p++; /* find the first non-space character */
      strcpy(frequency,e_p);
    } else if (strcmp(tag,"POLARISATION")==0){
      /* values are space delimited, so get all values */
      e_p=strstr(buf,"="); /* find the equals sign */
      while((*e_p==' ')||(*e_p=='='))
	e_p++; /* find the first non-space character */
      strcpy(polarisation,e_p);
    } else if (strcmp(tag,"SIDEBAND")==0){
      /* values are space delimited, so get all values */
      e_p=strstr(buf,"="); /* find the equals sign */
      while((*e_p==' ')||(*e_p=='='))
	e_p++; /* find the first non-space character */
      strcpy(sideband,e_p);
    } else if (strcmp(tag,"REFERENCEANT")==0){
      strcpy(referenceant,value);
    } else { /* unrecognised tag */
      (void)snprintf(tmp,BUFFSIZE,"Unrecognised tag in file %s: %s",full_path,tag);
      PrintStatus(tmp);
    }
  }
  fclose(profile);
  /* check that we have enough information for an experiment */
  if ((strlen(experiment_id)>0)&&
      ((strcmp(fringe_check,"yes")==0)||(strcmp(fringe_check,"no")==0))&&
      (strlen(channels)>0)&&
      (vsibmode!=0)&&
      (bandwidth!=0)&&
      (((strlen(time_start)>0)&&(imload==0))||(imload==1))&&
      (((strlen(time_end)>0)&&(imload==0))||(imload==1))){
    time(&time_now);
    if (imload==0){
      if ((sscanf(time_end,"%d:%d:%d %d/%d/%d",&enhour,&enminute,&ensecond,&enday,&enmonth,&enyear))!=6){
	strcpy(failure,PROF_ENTIME_INV_ERR);
	if (recovermode==1) return (NO_ERROR);
	else return(PROF_ENTIME_INV);
      }
      assign_time(enyear,enmonth,enday,enhour,enminute,ensecond,&end_time);
      if (end_time<time_now){
	/* the end time is in the past - we cannot load this profile */
	strcpy(failure,PROF_ENTIME_PAST_ERR);
	if (recovermode==1) return (NO_ERROR);
	else return(PROF_ENTIME_PAST);
      }
      if ((sscanf(time_start,"%d:%d:%d %d/%d/%d",&sthour,&stminute,&stsecond,&stday,&stmonth,&styear))!=6){
	strcpy(failure,PROF_STTIME_INV_ERR);
	if (recovermode==1) return (NO_ERROR);
	else return(PROF_STTIME_INV);
      }
      assign_time(styear,stmonth,stday,sthour,stminute,stsecond,&start_time);
      if (start_time>end_time){
	strcpy(failure,PROF_TIME_REVERSE_ERR);
	if (recovermode==1) return (NO_ERROR);
	else return(PROF_TIME_REVERSE);
      }
      if (start_time<time_now){
	/* there are three options here */
	/* 1 - the server crashed while this experiment was running, and the recorder
	   is still running on this experiment: we load the experiment, and take 
	   control of the recorder */
	/* 2 - the server crashed before this experiment started, and is now being 
	   restarted after the start time: we start this experiment, albeit late */
	/* 3 - this profile is being loaded now for the first time: we don't start 
	   this experiment, and return an error */
	if (recovermode!=1){ /* we're not being called to recover - option 3 */
	  strcpy(failure,PROF_STTIME_PAST_ERR);
	  return(PROF_STTIME_PAST);
	}
	if (status_server.is_recording==NO){ /* option 2 */
	  start_time=time_now+(time_t)(2*STATUS_UPDATE+EXPSWITCHTIME); /* start some time from now */
	  if (end_time<start_time){ /* we're too late basically */
	    strcpy(failure,PROFILE_NOSTART_ERR);
	    if (recovermode==1) return (NO_ERROR);
	    else return(PROFILE_NOSTART);
	  }
	} else { /* option 1 */
	  /* we load the recorder settings into a sandbox settings */
	  /* and then we check later that they match the experiment settings */
	  MALLOC(runSettings,sizeof(recorder));
	  if ((lres=LoadRecorderSettings(runSettings,failure))!=NO_ERROR){
	    if (recovermode==1) return (NO_ERROR);
	    else return(lres);
	  }
	}
      }
    }
    /* now we check that the data is valid */
    MALLOC(newSettings,sizeof(recorder));
    /* VSIB mode */
    (void)snprintf(tmp,BUFFSIZE,"vsib_mode=%d",vsibmode);
    if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
      free(newSettings);
      if (recovermode==1) return (NO_ERROR);
      else return(lres);
    }
    /* bandwidth */
    (void)snprintf(tmp,BUFFSIZE,"bandwidth=%d",bandwidth);
    if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
      free(newSettings);
      if (recovermode==1) return (NO_ERROR);
      else return(lres);
    }
    /* compression */
    (void)snprintf(tmp,BUFFSIZE,"compression=%s",channels);
    if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
      free(newSettings);
      if (recovermode==1) return (NO_ERROR);
      else return(lres);
    }
    /* fringe check */
    if (strcmp(fringe_check,"yes")==0){
      strcpy(tmp,"fringetest=yes");
    } else {
      strcpy(tmp,"fringetest=no");
    }
    if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
      free(newSettings);
      if (recovermode==1) return (NO_ERROR);
      else return(lres);
    }

    /* if we've been asked to load immediately, record time is not set */
    if (imload==1){
      if ((lres=LoadSettings(newSettings,RESET_RECORD_TIME,failure))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }

    /* that's the required information */
    /* now for the optional ones */
    /* the recording disk */
    if (strlen(required_disk)>0){
      if ((strlen(required_recorder)>0)&&
	  (strcmp(required_recorder,"local")!=0)){
	/* need to specify the remote recorder first */
	(void)snprintf(tmp,BUFFSIZE,"remrecorder=%s",required_recorder);
	if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	  free(newSettings);
	  if (recovermode==1) return (NO_ERROR);
	  else return(lres);
	}
      }
      (void)snprintf(tmp,BUFFSIZE,"recordingdisk=%s",required_disk);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
      newSettings->auto_disk_select=AUTODISK_DISABLED;
    } else {
      /* determine the best disk to use later */
      newSettings->auto_disk_select=AUTODISK_ANY;
      newSettings->recordingdisk=NULL;
    }
    if ((strlen(required_recorder)>0)&&
	(strlen(required_disk)==0)){
      (void)snprintf(tmp,BUFFSIZE,"remrecorder=%s",required_recorder);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
      /* if it's a local recorder, then we switch auto disk mode
         into local, otherwise switch it to remote auto disk */
      if ((strcmp(required_recorder,"local")==0)&&
	  (newSettings->auto_disk_select!=AUTODISK_DISABLED)){
	newSettings->auto_disk_select=AUTODISK_LOCAL;
      } else {
	if (newSettings->recordingdisk==NULL){
	  newSettings->auto_disk_select=AUTODISK_REMOTE;
	}
      }
    }
    /* the directory name */
    if (strlen(directory_name)>0){
      (void)snprintf(tmp,BUFFSIZE,"directory_name=%s",directory_name);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    } else { /* not specified - defaults to the experiment id */
      (void)snprintf(tmp,BUFFSIZE,"directory_name=%s",experiment_id);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }
    /* the filename prefix */
    if (strlen(filename)>0){
      (void)snprintf(tmp,BUFFSIZE,"filename_prefix=%s",filename);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    } else { /* not specified - defaults to the experiment id + the station name */
      (void)snprintf(tmp,BUFFSIZE,"filename_prefix=%s_%s",experiment_id,STATION_NAME);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }
    /* 10s starts */
    if (strlen(round_start)>0){
      if (strcmp(round_start,"yes")==0){
	/* restrict start to a 10s boundary */
	strcpy(tmp,"round_start=on");
      } else if (strcmp(round_start,"no")==0){
	/* start anytime - I don't expect to see this in a profile,
	   since there's already a 10% chance you will start on a
	   10s boundary, so why bother specifying that you don't care
	   when the recording starts */
	strcpy(tmp,"round_start=off");
      } else {
	/* unrecognised value */
	free(newSettings);
	strcpy(failure,EXP_ROUNDED_BAD_ERR);
	if (recovermode==1) return (NO_ERROR);
	else return(EXP_ROUNDED_BAD);
      }
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }
    /* VSIB device */
    if (strlen(vsib_device)>0){
      (void)snprintf(tmp,BUFFSIZE,"vsib_device=%s",vsib_device);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    } else {
      /* reset to default */
      strcpy(tmp,"reset=vsib_device");
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }
    /* file size */
    if (strlen(filesize)>0){
      (void)snprintf(tmp,BUFFSIZE,"filesize=%s",filesize);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    } else {
      /* reset to default */
      strcpy(tmp,"reset=filesize");
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }
    /* block size */
    if (blocksize!=-1){
      (void)snprintf(tmp,BUFFSIZE,"blocksize=%d",blocksize);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    } else {
      /* reset to default */
      strcpy(tmp,"reset=blocksize");
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }
    /* clock rate */
    if (clockrate!=-1){
      (void)snprintf(tmp,BUFFSIZE,"clockrate=%d",clockrate);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    } else {
      /* reset to default */
      strcpy(tmp,"reset=clockrate");
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }
    /* Mark5B recording */
    if (strlen(mark5b)>0){
      (void)snprintf(tmp,BUFFSIZE,"mark5b=%s",mark5b);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    } else {
      /* reset to default */
      strcpy(tmp,"reset=mark5b");
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }
    /* UDP */
    if (udp!=0){
      (void)snprintf(tmp,BUFFSIZE,"udp=%d",udp);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    } else {
      /* reset to default */
      strcpy(tmp,"reset=udp");
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }

    /* the header variables */
    /* NUMBITS */
    if (numbits!=-1){
      (void)snprintf(tmp,BUFFSIZE,"numbits=%d",numbits);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }
    /* ENCODING */
    if (strlen(encoding)>0){
      (void)snprintf(tmp,BUFFSIZE,"encoding=%s",encoding);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }
    /* FREQUENCY */
    if (strlen(frequency)>0){
      /* need to replace spaces with underscores for the data routine */
      while((s_p=strstr(frequency," "))!=NULL)
	*s_p='_';
      (void)snprintf(tmp,BUFFSIZE,"frequency=%s",frequency);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }
    /* POLARISATION */
    if (strlen(polarisation)>0){
      while((s_p=strstr(polarisation," "))!=NULL)
	*s_p='_';
      (void)snprintf(tmp,BUFFSIZE,"polarisation=%s",polarisation);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }
    /* SIDEBAND */
    if (strlen(sideband)>0){
      while((s_p=strstr(sideband," "))!=NULL)
	*s_p='_';
      (void)snprintf(tmp,BUFFSIZE,"sideband=%s",sideband);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }
    /* REFERENCEANT */
    if (strlen(referenceant)>0){
      (void)snprintf(tmp,BUFFSIZE,"referenceant=%s",referenceant);
      if ((lres=data_handler(tmp,failure,newSettings))!=NO_ERROR){
	free(newSettings);
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
    }

    /* now we check whether the sandbox settings we loaded when asked to recover */
    /* match the settings the current recorder task is using */
    if (runSettings!=NULL){
      if ((strcmp(newSettings->directory_name,runSettings->directory_name)!=0)||
	  (strcmp(newSettings->filename_prefix,runSettings->filename_prefix)!=0)||
	  (strcmp(newSettings->compression,runSettings->compression)!=0)||
	  (newSettings->vsib_mode!=runSettings->vsib_mode)||
	  (newSettings->bandwidth!=runSettings->bandwidth)){
	/* not the experiment which is being recorded right now */
	free(runSettings);
	free(newSettings);
	strcpy(failure,PROF_NOTCURRENT_ERR);
	if (recovermode==1) return (NO_ERROR);
	else return(PROF_NOTCURRENT);
      }
      /* they've matched, they're the current settings */
      status_server.current_settings=runSettings;
      /* set some status variables */
      status_server.recorder_end=end_time;
      status_server.experiment_mode=EXPERIMENT_QUEUE;
    }
      
    /* if we get here, all checks have been satisfied */
    /* we assign the time ourselves */
    if (imload==0){
      temp_time=gmtime(&start_time);
      (void)snprintf(newSettings->record_start_date,BUFFSIZE,"%4d%02d%02d",(temp_time->tm_year)+1900,(temp_time->tm_mon)+1,temp_time->tm_mday);
      (void)snprintf(newSettings->record_start_time,BUFFSIZE,"%02d%02d%02d",temp_time->tm_hour,temp_time->tm_min,temp_time->tm_sec);
    }

    if (imload==0){
      /* allocate a new experiment */
      MALLOC(newExperiment,sizeof(experiment));
/*       newExperiment=malloc(sizeof(experiment)); */
      strcpy(newExperiment->experiment_id,experiment_id);
      newExperiment->start_time=start_time;
      newExperiment->end_time=end_time;
      newExperiment->record_settings=newSettings;
      newExperiment->started=NO;
      /* place it in the queue */
      if ((lres=experiment_add(newExperiment,failure))!=NO_ERROR){
	if (recovermode==1) return (NO_ERROR);
	else return(lres);
      }
      /* and make a recorder target so the user can see the settings
         for this experiment immediately */
      copy_recordertarget(RECORDERTARGET_MAKE,experiment_id,newSettings,
			  failure); /* we don't actually care if this works */
    } else {
      /* just make the current settings the settings we just loaded */
      copy_recorder(newSettings,status_server.current_settings);
/*       status_server.current_settings=newSettings; */
      FREE(newSettings);
    }

    /* we're done */
    return(NO_ERROR);

  } else { /* some necessary info is missing */
    strcpy(failure,PROFILE_INVALID_ERR);
    if (recovermode==1) return (NO_ERROR);
    else return(PROFILE_INVALID);
  }
}

/*
  This routine takes the current settings and saves them as an experiment
  profile on disk.
*/
int SaveExperiment(char *expid,char *failure){
  char directory_name[BUFFSIZE],filename_prefix[BUFFSIZE],experiment_id[BUFFSIZE],fringecheck[BUFFSIZE];
  char channels[BUFFSIZE],tmp[BUFFSIZE],start_time[BUFFSIZE],end_time[BUFFSIZE];
  char encoding[BUFFSIZE],frequency[BUFFSIZE],polarisation[BUFFSIZE],sideband[BUFFSIZE];
  char referenceant[BUFFSIZE],*u_p,clean_time[BUFFSIZE],round_start[BUFFSIZE];
  char vsib_device[BUFFSIZE],filesize[BUFFSIZE],evlbi[BUFFSIZE];
  char remote_host[BUFFSIZE],mark5b[BUFFSIZE];
  int vsibmode=0,bandwidth=0,hour,minute,second,day,month,year,recordtime,numbits,ipd;
  int blocksize=0,clockrate=0,remote_port=0,tcp_window_size=0,udp=0;
  time_t tmp_time;
  FILE *profile=NULL;

  /* clear all strings */
  experiment_id[0]='\0';
  directory_name[0]='\0';
  filename_prefix[0]='\0';
  fringecheck[0]='\0';
  channels[0]='\0';
  start_time[0]='\0';
  end_time[0]='\0';
  clean_time[0]='\0';
  round_start[0]='\0';
  vsib_device[0]='\0';
  filesize[0]='\0';
  evlbi[0]='\0';
  remote_host[0]='\0';
  mark5b[0]='\0';

  /* first, check that we have all the information we need to make an */
  /* experiment profile */
  /* the directory name, or a supplied experiment id */
  if ((strlen(status_server.current_settings->directory_name)!=0)|| /* directory name is set */
      (strlen(expid)!=0)){ /* experiment id is given */
    if (strlen(expid)==0){ /* experiment id not supplied */
      if (status_server.current_settings->fringe_test==0){ /* not a fringe-test */
	strcpy(experiment_id,status_server.current_settings->directory_name);
	strcpy(fringecheck,"no");
      } else if (status_server.current_settings->fringe_test==1){
	/* we can't get the experiment id from the directory name */
	strcpy(failure,EXPSAVE_FCHECK_ERR);
	return(EXPSAVE_FCHECK);
      }
    } else { /* experiment id has been supplied */
      strcpy(experiment_id,expid);
      if (status_server.current_settings->fringe_test==1){
	strcpy(fringecheck,"yes");
      } else if (status_server.current_settings->fringe_test==0){
	strcpy(directory_name,status_server.current_settings->directory_name);
	strcpy(fringecheck,"no");
      }
      if (strcmp(directory_name,experiment_id)==0){
	/* no need to write directory name, LoadExperiment will figure it out */
	directory_name[0]='\0';
      }
    }
  } else {
    /* directory name is missing */
    strcpy(failure,EXPSAVE_DIRNAME_ERR);
    return(EXPSAVE_DIRNAME);
  }
      
  /* check for filename */
  if (strlen(status_server.current_settings->filename_prefix)!=0){
    (void)snprintf(tmp,BUFFSIZE,"%s_%s",experiment_id,STATION_NAME);
    if (strcmp(tmp,status_server.current_settings->filename_prefix)!=0){
      /* filename is not just default experiment name + the station name */
      strcpy(filename_prefix,status_server.current_settings->filename_prefix);
    }
  }

  /* check for clean time */
  if (strlen(status_server.current_settings->clean_time)!=0){
    strcpy(clean_time,status_server.current_settings->clean_time);
  }

  /* check for 10s starts */
  if (status_server.rounded_start==ROUNDSTART_YES){
    /* if currently set, then set it explicitly in the profile */
    /* if not currently set, what does it matter what the user chooses when 
       the profile is loaded? */
    strcpy(round_start,"yes");
  }

  /* check for VSIB device */
  if (strcmp(status_server.current_settings->vsib_device,
	     DEFAULT_DEVICE)!=0){
    /* if set different to default, set explicitly in the profile */
    strcpy(vsib_device,status_server.current_settings->vsib_device);
  }

  /* check for filesize */
  if ((status_server.current_settings->filesize_or_time!=DEFAULT_FILESIZE)||
      (status_server.current_settings->filesize_is_time!=DEFAULT_FILETIME)){
    (void)snprintf(filesize,BUFFSIZE,"%d",
		   (int)status_server.current_settings->filesize_or_time);
    if (status_server.current_settings->filesize_is_time==YES)
      strcat(filesize,"s");
  }

  /* check for blocksize */
  if (status_server.current_settings->blocksize!=DEFAULT_BLOCKSIZE)
    blocksize=(int)status_server.current_settings->blocksize;

  /* check for clockrate */
  if (status_server.current_settings->clockrate!=DEFAULT_CLOCKRATE)
    clockrate=(int)status_server.current_settings->clockrate;

  /* check for Mark5B recording */
  if (status_server.current_settings->mark5b_enabled!=DEFAULT_MARK5B){
    if (status_server.current_settings->mark5b_enabled==YES)
      strcpy(mark5b,"on");
    else if (status_server.current_settings->mark5b_enabled==NO)
      strcpy(mark5b,"off");
  }

  /* copy some required information */
  strcpy(channels,status_server.current_settings->compression); /* compression mode */
  vsibmode=status_server.current_settings->vsib_mode; /* the vsib mode */
  bandwidth=status_server.current_settings->bandwidth; /* the observing bandwidth */

  /* copy the header info */
  numbits=status_server.current_settings->numbits;
  ipd=status_server.current_settings->ipd;
  strcpy(encoding,status_server.current_settings->encoding);
  strcpy(frequency,status_server.current_settings->frequency);
  strcpy(polarisation,status_server.current_settings->polarisation);
  strcpy(sideband,status_server.current_settings->sideband);
  strcpy(referenceant,status_server.current_settings->referenceant);
  
  /* now look for the start and end times */
  if (strlen(status_server.current_settings->record_start_time)!=0){
    /* there is a start time/date, we need to put that in */
    sscanf(status_server.current_settings->record_start_date,"%4d%2d%2d",&year,&month,&day);
    sscanf(status_server.current_settings->record_start_time,"%2d%2d%2d",&hour,&minute,&second);
    assign_time(year,month,day,hour,minute,second,&tmp_time);
    timeasstring(tmp_time,tmp);
    strcpy(start_time,tmp);
    recordtime=-1;
    recordtime=timeinseconds(status_server.current_settings->record_time);
    if (recordtime>0){
      timeasstring((tmp_time+(time_t)recordtime),tmp);
      strcpy(end_time,tmp);
    } else {
      strcpy(failure,RECORDING_TIME_ERR);
      return(RECORDING_TIME);
    }
  } else {
    /* this a profile which can be used any time in manual mode */
    strcpy(start_time,"immediate");
    strcpy(end_time,"immediate");
  }

  /* we have all the information, now we try to write the experiment profile */
  /* determine first whether there is already a file with the same name */
  (void)snprintf(tmp,BUFFSIZE,"%s/%s",experiment_location,experiment_id);
  if ((profile=fopen(tmp,"r"))!=NULL){
    /* already exists */
    fclose(profile);
    strcpy(failure,EXPSAVE_EXISTS_ERR);
    return(EXPSAVE_EXISTS);
  }
  if ((profile=fopen(tmp,"w"))==NULL){
    /* unable to create files in the profile directory */
    strcpy(failure,EXPSAVE_NOWRITE_ERR);
    return(EXPSAVE_NOWRITE);
  }
  fprintf(profile,"EXPERIMENT_ID = %s\n",experiment_id);
  fprintf(profile,"TIME_START = %s\n",start_time);
  fprintf(profile,"TIME_END   = %s\n",end_time);
  fprintf(profile,"FRINGE_CHECK = %s\n",fringecheck);
  fprintf(profile,"CHANNELS = %s\n",channels);
  fprintf(profile,"VSIBMODE = %d\n",vsibmode);
  fprintf(profile,"BANDWIDTH = %d\n",bandwidth);
  if (strlen(directory_name)!=0)
    fprintf(profile,"DIRECTORY_NAME = %s\n",directory_name);
  if (strlen(filename_prefix)!=0)
    fprintf(profile,"FILENAME = %s\n",filename_prefix);
  if (strlen(clean_time)!=0)
    fprintf(profile,"CLEAN_TIME = %s\n",clean_time);
  if (strlen(round_start)!=0)
    fprintf(profile,"ROUND_START = %s\n",round_start);
  if (strlen(vsib_device)!=0)
    fprintf(profile,"VSIB_DEVICE = %s\n",vsib_device);
  if (strlen(filesize)!=0)
    fprintf(profile,"FILESIZE = %s\n",filesize);
  if (blocksize!=0)
    fprintf(profile,"BLOCKSIZE = %d\n",blocksize);
  if (clockrate!=0)
    fprintf(profile,"CLOCKRATE = %d\n",clockrate);
  if (strlen(evlbi)!=0){
    fprintf(profile,"EVLBI_MODE = %s\n",evlbi);
    if (strlen(remote_host)!=0)
      fprintf(profile,"REMOTE_HOST = %s\n",remote_host);
    if (remote_port!=0)
      fprintf(profile,"REMOTE_PORT = %d\n",remote_port);
    if (tcp_window_size!=0)
      fprintf(profile,"TCP_WINDOW_SIZE = %d\n",tcp_window_size);
    if (udp!=0)
      fprintf(profile,"UDP = %d\n",udp);
  }
  if (strlen(mark5b)!=0)
    fprintf(profile,"MARK5B = %s\n",mark5b);
  /* note that we aren't writing any required disk - we'll let the experiment control */
  /* figure that out for us, or let the user add it in themselves */
  if (numbits!=DEFAULT_NUMBITS)
    fprintf(profile,"NUMBITS = %d\n",numbits);
  if (ipd!=DEFAULT_IPD)
    fprintf(profile,"IPD = %d\n",ipd);
  if (strcmp(encoding,DEFAULT_ENCODING)!=0){
    /* change spaces to underscores */
    while((u_p=strstr(encoding," "))!=NULL)
      *u_p='_';
    fprintf(profile,"ENCODING = %s\n",encoding);
  }
  if (strcmp(frequency,DEFAULT_FREQUENCY)!=0){
    while((u_p=strstr(frequency," "))!=NULL)
      *u_p='_';
    fprintf(profile,"FREQUENCY = %s\n",frequency);
  }
  if (strcmp(polarisation,DEFAULT_POLARISATION)!=0){
    while((u_p=strstr(polarisation," "))!=NULL)
      *u_p='_';
    fprintf(profile,"POLARISATION = %s\n",polarisation);
  }
  if (strcmp(sideband,DEFAULT_SIDEBAND)!=0){
    while((u_p=strstr(sideband," "))!=NULL)
      *u_p='_';
    fprintf(profile,"SIDEBAND = %s\n",sideband);
  }
  if (strcmp(referenceant,DEFAULT_REFERENCEANT)!=0)
    fprintf(profile,"REFERENCEANT = %s\n",referenceant);
  fclose(profile);

  /* all done */
  return(NO_ERROR);

}

/*
  This routine is used to load the recorder settings if the server crashed
  while the recorder was running. This is so we can recover gracefully without
  needing to restart the recording process, or having to resort to manual
  control.
*/
int LoadRecorderSettings(recorder *settings,char *failmsg){
  FILE *recorder_log=NULL;
  char settings_line[BUFFSIZE],paramname[BUFFSIZE],paramcval[BUFFSIZE],passdata[BUFFSIZE],failure[BUFFSIZE];
  char buf[BUFFSIZE],*p=NULL;
  int i,numparams,paramival=-1,j;

  if (status_server.is_recording==NO){
    strcpy(failmsg,LOAD_NOT_RUNNING_ERR);
    return(LOAD_NOT_RUNNING);
  }
  (void)snprintf(buf,BUFFSIZE,"%s/recorder_%d_settings",tmp_directory,status_server.recorder_pid);
  printf("loading recorder settings file %s\n",buf);
  if ((recorder_log=fopen(buf,"r"))==NULL){ /* the file isn't there */
    strcpy(failmsg,FILE_UNREADABLE_ERR);
    return(FILE_UNREADABLE);
  }

  /* load the recorder settings into the recorder structure */
  numparams=26; /* there are 25 parameters to read in */
  for (i=0;i<numparams;i++){
/*     printf("i = %d\n",i); */
    for (j=0;j<BUFFSIZE;j++){
      settings_line[j]='\0';
    }
    if ((fgets(settings_line,BUFFSIZE,recorder_log))==NULL){ /* shouldn't get an EOF error! */
      strcpy(failmsg,UNEXPECTED_EOF_ERR);
      return(UNEXPECTED_EOF);
    }
    switch (i){
    case 0: case 1: case 2: case 3: case 4: case 5: case 8:
    case 11: case 12: case 15: case 16: case 19: case 21: case 22:
    case 23: case 24: case 25:
      sscanf(settings_line,"%s %s\n",paramname,paramcval);
      (void)snprintf(passdata,BUFFSIZE,"%s=%s",paramname,paramcval);
      break;
    case 6: case 7: case 13: case 14: case 17: case 18:
    case 20:
      sscanf(settings_line,"%s %d\n",paramname,&paramival);
      (void)snprintf(passdata,BUFFSIZE,"%s=%d",paramname,paramival);
      break;
    case 9:
      strcpy(paramcval,settings_line+20);
      if ((p=strchr(paramcval,'\n'))!=NULL)
	*p='\0'; /* remove the newline */
      break;
    case 10:
      sscanf(settings_line,"%s %d\n",paramname,&paramival);
      if (paramival==1){ /* this is a fringe-test */
	(void)snprintf(passdata,BUFFSIZE,"%s=yes",paramname);
      } else if (paramival==0){ /* not a fringe-test */
	(void)snprintf(passdata,BUFFSIZE,"%s=no",paramname);
      } else {
	strcpy(failmsg,CORRUPT_SETTINGS_ERR);
	return(CORRUPT_SETTINGS);
      }
    }
    if ((i!=1)&&(i!=2)&&(i!=9)&&(i<11)){
      if ((data_handler(passdata,failure,settings))!=NO_ERROR){
	strcpy(failmsg,CORRUPT_SETTINGS_ERR);
	return(CORRUPT_SETTINGS);
      }
    } else { /* we don't want to check the start time and date - 
		it is possible the recorder has already started */
      if (i==1){
	if (strcmp(paramcval,"-1")==0){
	  settings->record_start_date[0]='\0';
	} else {
	  strcpy(settings->record_start_date,paramcval);
	}
      } else if (i==2) {
	if (strcmp(paramcval,"-1")==0){
	  settings->record_start_time[0]='\0';
	} else {
	  strcpy(settings->record_start_time,paramcval);
	}
      } else if (i==9){ /* the command used to start the recorder */
	strcpy(status_server.recorder_command,paramcval);
      } else if (i>=11){
	/* might get an error while entering header information, but */
	/* we don't care */
	data_handler(passdata,failure,settings);
      }
    }
  }
  fclose(recorder_log);

  /* now we set the recording disk and directory status variables */
  if (status_server.is_recording==YES){
    status_server.recording_to_disk=status_server.current_settings->recordingdisk;
    strcpy(status_server.recording_path,status_server.current_settings->directory_name);
  }

  /* load successful */
  return(NO_ERROR);

}

/*
  This routine is called by the recorder_control routine after it issues
  the recorder start command. It is supposed to write out all the data 
  relevant to the recorder, so it can be controlled even after a server
  restart.
*/
int WriteRecorderSettings(recorder *settings,char *failure){
  FILE *recorder_log=NULL;
  char recordfile[BUFFSIZE],*s_p,tmp[BUFFSIZE];

  if (status_server.is_recording==NO){ /* not recording */
    strcpy(failure,RECORDER_OFF_ERR);
    return(RECORDER_OFF);
  }
  /* we don't bother checking if the file exists, only if we can open it for writing */
  (void)snprintf(recordfile,BUFFSIZE,"%s/recorder_%d_settings",tmp_directory,status_server.recorder_pid);
  if ((recorder_log=fopen(recordfile,"w"))==NULL){
    strcpy(failure,FILE_UNWRITABLE_ERR);
    return(FILE_UNWRITABLE);
  }
  
  fprintf(recorder_log,"%-17s   %s\n","record_time",settings->record_time);
  if (strlen(settings->record_start_date)!=0){
    fprintf(recorder_log,"%-17s   %s\n","record_start_date",settings->record_start_date);
    fprintf(recorder_log,"%-17s   %s\n","record_start_time",settings->record_start_time);
  } else {
    fprintf(recorder_log,"%-17s   %d\n","record_start_date",-1);
    fprintf(recorder_log,"%-17s   %d\n","record_start_time",-1);
  }
  fprintf(recorder_log,"%-17s   %s\n","directory_name",settings->directory_name);
  fprintf(recorder_log,"%-17s   %s\n","filename_prefix",settings->filename_prefix);
  fprintf(recorder_log,"%-17s   %s\n","compression",settings->compression);
  fprintf(recorder_log,"%-17s   %d\n","vsib_mode",settings->vsib_mode);
  fprintf(recorder_log,"%-17s   %d\n","bandwidth",settings->bandwidth);
  fprintf(recorder_log,"%-17s   %s\n","recordingdisk",settings->recordingdisk->diskpath);
  fprintf(recorder_log,"%-17s   %s\n","recordercommand",status_server.recorder_command);
  fprintf(recorder_log,"%-17s   %d\n","fringetest",settings->fringe_test);
  fprintf(recorder_log,"%-17s   %s\n","vsib_device",settings->vsib_device);
  (void)snprintf(tmp,BUFFSIZE,"%d",(int)settings->filesize_or_time);
  if (settings->filesize_is_time==YES)
    strcat(tmp,"s");
  fprintf(recorder_log,"%-17s   %s\n","filesize",tmp);
  fprintf(recorder_log,"%-17s   %d\n","blocksize",(int)settings->blocksize);
  fprintf(recorder_log,"%-17s   %d\n","clockrate",(int)settings->clockrate);
  if (settings->mark5b_enabled==YES)
    strcpy(tmp,"on");
  else
    strcpy(tmp,"off");
  fprintf(recorder_log,"%-17s   %s\n","mark5b",tmp);
  fprintf(recorder_log,"%-17s   %d\n","udp",settings->udp_enabled);
  fprintf(recorder_log,"%-17s   %d\n","numbits",settings->numbits);
  if (settings->ipd) 
    fprintf(recorder_log,"%-17s   %d\n","ipd",settings->ipd);
  fprintf(recorder_log,"%-17s   %s\n","encoding",settings->encoding);
  strcpy(tmp,settings->frequency);
  while((s_p=strstr(tmp," "))!=NULL)
    *s_p='_';
  fprintf(recorder_log,"%-17s   %s\n","frequency",tmp);
  strcpy(tmp,settings->polarisation);
  while((s_p=strstr(tmp," "))!=NULL)
    *s_p='_';
  fprintf(recorder_log,"%-17s   %s\n","polarisation",tmp);
  strcpy(tmp,settings->sideband);
  while((s_p=strstr(tmp," "))!=NULL)
    *s_p='_';
  fprintf(recorder_log,"%-17s   %s\n","sideband",tmp);
  fprintf(recorder_log,"%-17s   %s\n","referenceant",settings->referenceant);
  fclose(recorder_log);
  return(NO_ERROR);

}


/*
  This routine is called to see if a process is still running, by
  checking for the PID number that we were given when it started,
  and checking that that PID is actually the process we think it is
*/
int checkrunning(int pid,char *pname){
  struct stat check_dir;
  int dir_check=-1,sysretval;
  char buf[BUFFSIZE],fbuf[BUFFSIZE];
  FILE *grepres=NULL;

  /* check for the existence of the directory PID in the proc fs */
  (void)snprintf(buf,BUFFSIZE,"/proc/%d",pid); /* the name of the directory */
/*   printf("[checkrunning] checking for directory %s\n",buf); */
  if ((dir_check=stat(buf,&check_dir))==0){ /* something with the directory name already exists */
    if (S_ISDIR(check_dir.st_mode)){
      /* there is a process with that PID */
      /* is is what we think it is? */
      (void)snprintf(buf,BUFFSIZE,"grep %s /proc/%d/cmdline > %s/grepres",pname,pid,tmp_directory);
/*       printf("[checkrunning] running grep command %s\n",buf); */
      sysretval=system(buf);
      (void)snprintf(buf,BUFFSIZE,"%s/grepres",tmp_directory);
      if ((grepres=fopen(buf,"r"))==NULL){
/* 	printf("[checkrunning] couldn't open grepresult\n"); */
	return(0); /* we'll assume the process was transient and is not the one we want */
      }
      if ((fgets(fbuf,BUFFSIZE,grepres))==NULL){
	/* the grep did not match anything - the process is something else */
/* 	printf("[checkrunning] nothing in grepres\n"); */
	fclose(grepres);
	remove(buf);
	return(0);
      } else {
	/* the grep did match the process name, it is what we think it is */
/* 	printf("[checkrunning] something in grepres\n"); */
	fclose(grepres);
	remove(buf);
	return(1);
      }
    } else {
/*       printf("[checkrunning] couldn't find directory\n"); */
      /* the process is no longer running */
      return(0);
    }
  } else {
/*     printf("[checkrunning] couldn't find anything with that name\n"); */
    return(0);
  }
}

/*
  This routine turns a date into a time_t value, so it can easily be compared
  to another time.
*/
void assign_time(int year,int month,int day,int hour,int minute,int second,time_t *t_assigned){

  struct tm time_st;
  time_st.tm_year=year-1900;
  time_st.tm_mon=month-1;
  time_st.tm_mday=day;
  time_st.tm_hour=hour;
  time_st.tm_min=minute;
  time_st.tm_sec=second;
  *t_assigned=timegm(&time_st);

}

/*
  This routine returns the current GMT as a string HH:MM:SS dd/mm/yyyy
  It is called by the logger routine for timestamp information
*/
void thetimenow(char *time_string){
  time_t time_now;
  struct tm *now_time;

  time(&time_now);
  now_time=gmtime(&time_now);
  (void)snprintf(time_string,BUFFSIZE,"%02d:%02d:%02d %02d/%02d/%4d",
	  now_time->tm_hour,now_time->tm_min,now_time->tm_sec,now_time->tm_mday,(now_time->tm_mon)+1,(now_time->tm_year)+1900);

}

/* 
   This routine sets the default socket options that we use so that clients
   cannot cause the server to stop responding if the client is malicious or
   badly programmed. It should be called after opening the socket.
*/
void socket_options(int sock,int call_flag){
  struct linger lingset;
  struct timeval timeouts;
  int reuseopt=1;

  /* set the TCP close options for the socket */
  /* we make the wait time 0 for socket closures so */
  /* when we ask the control forks to close sockets, they */
  /* do so immediately and do not disrupt control in case */
  /* of a server crash */
  lingset.l_onoff=1;
  lingset.l_linger=0;
  /* set timeout options for the socket */
  /* we allow the connection to stay open for 1 second before */
  /* exiting with an error, so that a malicious client can't make */
  /* the server stop responding for very long */
  timeouts.tv_sec=1;

  /* now apply the options to the socket */
  setsockopt(sock,SOL_SOCKET,SO_LINGER,&lingset,(socklen_t)(sizeof(struct linger)));
  /* and the socket timeout option */
  setsockopt(sock,SOL_SOCKET,SO_SNDTIMEO,&timeouts,(socklen_t)(sizeof(struct timeval)));
  setsockopt(sock,SOL_SOCKET,SO_REUSEADDR,(char*)&reuseopt,sizeof(reuseopt));
  if (call_flag==SOCKETOPTIONS_OTHER){
    /* we only set a receive timeout for sockets other than the */
    /* main listener. if the main listener has a receive timeout, */
    /* it doesn't actually listen */
    setsockopt(sock,SOL_SOCKET,SO_RCVTIMEO,&timeouts,(socklen_t)(sizeof(struct timeval)));
  }

  /* make the socket stay with this instance only, and do not transfer to our children */
  fcntl(sock,F_SETFD,fcntl(sock,F_GETFD)|FD_CLOEXEC);

}

/*
  This routine removes a remote recorder from the list.
*/
void RemoveRemote(remoterecorder *badrecorder,int removeflag){
  remoterecorder *cyclerecorder=NULL;
  outputdisk *cycledisk=NULL,*freedisk=NULL;
  char tmp[BUFFSIZE];

  snprintf(tmp,BUFFSIZE,"Removing remote recorder %s from our list.",
	   badrecorder->commonname);
  PrintStatus(tmp);

  if (removeflag==REMOTEHOST_REMOVE_COMMSFAIL){
    /* we've had a failure of communication with this host, so increment
       the communication failure counter */
    badrecorder->comms_failures++;
    /* we don't remove a host before getting too many communications
       failures */
    if (badrecorder->comms_failures<=MAX_COMMS_FAIL){
      badrecorder->comms_wait=YES;
      return;
    }
  }

  /* some safety checks */
  /* are we recording to this remote recorder? */
  if (status_server.is_recording==YES){
    /* we're recording, check if it's to this recorder */
    if (status_server.recording_settings->targetrecorder==badrecorder){
      /* we need to stop the recording */
      recorder_control(RECSTOP,status_server.recording_settings,tmp);
    }
  }
  /* are we receiving from this remote recorder? */
  if (badrecorder->receiver!=NULL){
    /* we need to stop the receiver */
    receiver_control(RECEIVER_STOP,badrecorder->commonname,status_server.current_settings,tmp);
  }

  /* close the socket */
  if (badrecorder->evlbi_enabled==NO){
    close(badrecorder->connectionsocket);
  }

  /* are we the first/only recorder? */
  if (status_server.remote_recorders==badrecorder){
    status_server.remote_recorders=badrecorder->next;
  }
  /* do some reconnections */
  for (cyclerecorder=status_server.remote_recorders;cyclerecorder!=NULL;
       cyclerecorder=cyclerecorder->next){
    if (cyclerecorder->next==badrecorder){
      cyclerecorder->next=badrecorder->next;
    }
  }
  /* free our memory */
  cycledisk=badrecorder->available_disks;
  while(cycledisk!=NULL){
    freedisk=cycledisk;
    cycledisk=cycledisk->next;
    FREE(freedisk);
  }
  free_hostent(badrecorder->host);
  FREE(badrecorder->host);
  FREE(badrecorder);
}

int AddRemote(char *message,remoterecorder *newrecorder){
  int msglen,received=0,timeremaining;
  char buf[BUFFSIZE],tmp[BUFFSIZE];
  struct sockaddr_in mesgclient;

  PrintStatus("adding new remote recorder");
  /* set an alarm - should we not be back within 2 seconds, we
     go directly to UpdateStatus */
  hangingout=YES;
  (void)signal(SIGALRM,UpdateStatusSignal);
  timeremaining=alarm(2);

  snprintf(tmp,BUFFSIZE,"connecting to %s @ %s on port %d\n",
	   newrecorder->commonname,newrecorder->hostname,
	   newrecorder->recorderserver_port);
/*   PrintStatus(tmp); */
  snprintf(tmp,BUFFSIZE,"sending message %s\n",message);
/*   PrintStatus(tmp); */
  
  if ((outsock=socket(PF_INET,SOCK_STREAM,IPPROTO_TCP))<0){
    PrintStatus("Unable to create socket - this is a server error!");
    return(GENERAL_ERROR);
  }

  snprintf(tmp,BUFFSIZE,"creating socket %d for remote host communications",
	   outsock);
  /*   PrintStatus(tmp); */
  //socket_options(outsock,SOCKETOPTIONS_OTHER);

  memset(&mesgclient,0,sizeof(mesgclient));
  mesgclient.sin_family=AF_INET;
  bcopy((char*)newrecorder->host->h_addr,(char*)&mesgclient.sin_addr.s_addr,
	newrecorder->host->h_length);
  mesgclient.sin_port=htons(newrecorder->recorderserver_port);
  if (connect(outsock,(struct sockaddr *)&mesgclient,sizeof(mesgclient))<0){
    PrintStatus("Unable to connect to client!");
    close(outsock);
    (void)alarm(timeremaining);
    hangingout=NO;
    return(GENERAL_ERROR);
  }
  msglen=(int)strlen(message);
  if (send(outsock,message,msglen,0)!=msglen){
    PrintStatus("Could not send message to client.");
    close(outsock);
    (void)alarm(timeremaining);
    hangingout=NO;
    return(GENERAL_ERROR);;
  }
  if ((received=recv(outsock,buf,BUFFSIZE-1,0))<1){
    PrintStatus("Did not receive acknowledgement from client.");
    close(outsock);
    (void)alarm(timeremaining);
    hangingout=NO;
    return(GENERAL_ERROR);
  } else {
    snprintf(tmp,BUFFSIZE,"response: %s\n",buf);
/*     PrintStatus(tmp); */
    if (strcmp(buf,"<succ />")!=0){
      strcpy(message,buf);
      PrintStatus("Client did not properly acknowledge message.");
      close(outsock);
      (void)alarm(timeremaining);
      hangingout=NO;
      return(GENERAL_ERROR);
    }
  }
  
  /* now the socket is to remain open */
  newrecorder->connectionsocket=outsock;
  snprintf(tmp,BUFFSIZE,"using socket %d for remote host communication",
	   newrecorder->connectionsocket);
  PrintStatus(tmp);
  outsock=0;
  (void)alarm(timeremaining);
  hangingout=NO;

  return(NO_ERROR);
}

int PushRemote(char *message,remoterecorder *sendrecorder){
  int msglen,received=0;/* ,timeremaining; */
  char buf[BUFFSIZE],tmp[BUFFSIZE];
/*   struct sockaddr_in mesgclient; */

  if (sendrecorder->comms_wait==YES){
    return(NO_ERROR);
  }
  snprintf(tmp,BUFFSIZE,"sending to %s @ %s on port %d, using socket %d \n",
	   sendrecorder->commonname,sendrecorder->hostname,
	   sendrecorder->recorderserver_port,sendrecorder->connectionsocket);
  PrintStatus(tmp);
  snprintf(tmp,BUFFSIZE,"sending message %s\n",message);
/*   PrintStatus(tmp); */
  
  msglen=(int)strlen(message);
  if (send(sendrecorder->connectionsocket,message,msglen,0)!=msglen){
    PrintStatus("Could not send message to client.");
    return(GENERAL_ERROR);;
  }
  if ((received=recv(sendrecorder->connectionsocket,buf,BUFFSIZE-1,0))<1){
    PrintStatus("Did not receive acknowledgement from client.");
    return(GENERAL_ERROR);
  } else {
    snprintf(tmp,BUFFSIZE,"response: %s\n",buf);
/*     PrintStatus(tmp); */
    if (strcmp(buf,"<succ />")!=0){
      strcpy(message,buf);
      PrintStatus("Client did not properly acknowledge message.");
      return(GENERAL_ERROR);
    }
  }

  sendrecorder->comms_failures=0;
  return(NO_ERROR);

}

/*
  This routine allows messages to be "pushed" to network clients that
  have registered to get such messages. This is so warnings and experiment
  commencement/ending messages can be sent to clients when they occur, not
  just when they are requested.
*/
void PushStatus(char *message){
  registered_clients *cycle_clients=NULL,*old_client=NULL,*dummy_client=NULL;
  int msglen,received=0,i,timeremaining;
  char buf[BUFFSIZE];
  struct sockaddr_in mesgclient;

  /* set an alarm - should we not be back within 2 seconds, we
     go directly to UpdateStatus */
  hangingout=YES;
  timeremaining=alarm(2);

  (void)snprintf(buf,BUFFSIZE,"Sending message to registered clients:\n%s",message);
  PrintStatus(buf);
  i=0;
  /* make a "dummy client" which can be used to point to the new head */
  /* should the old head fail */
  MALLOCV(dummy_client,sizeof(registered_clients));
  dummy_client->next=message_clients;
  old_client=dummy_client;
  for (cycle_clients=message_clients;cycle_clients!=NULL;cycle_clients=cycle_clients->next){
    i++;
    (void)snprintf(buf,BUFFSIZE,"Client %d: %s:%d",i,cycle_clients->ip_address,cycle_clients->port);
/*     PrintStatus(buf); */
    if ((outsock=socket(PF_INET,SOCK_STREAM,IPPROTO_TCP))<0){
      PrintStatus("Unable to create socket - this is a server error!");
      (void)alarm(timeremaining);
      hangingout=NO;
      return;
    }
    socket_options(outsock,SOCKETOPTIONS_OTHER);

    memset(&mesgclient,0,sizeof(mesgclient));
    mesgclient.sin_family=AF_INET;
    mesgclient.sin_addr.s_addr=inet_addr(cycle_clients->ip_address);
    mesgclient.sin_port=htons(cycle_clients->port);
    if (connect(outsock,(struct sockaddr *)&mesgclient,sizeof(mesgclient))<0){
      PrintStatus("Unable to connect to client!");
      close(outsock);
      (void)alarm(timeremaining);
      hangingout=NO;
      RemoveClient(cycle_clients->ip_address,
		   cycle_clients->port);
      cycle_clients=old_client;
      dummy_client->next=message_clients;
      continue;
    }
    msglen=(int)strlen(message);
    if (send(outsock,message,msglen,0)!=msglen){
      PrintStatus("Could not send message to client.");
      close(outsock);
      (void)alarm(timeremaining);
      hangingout=NO;
      RemoveClient(cycle_clients->ip_address,
		   cycle_clients->port);
      cycle_clients=old_client;
      dummy_client->next=message_clients;
      continue;
    }
    if ((received=recv(outsock,buf,BUFFSIZE-1,0))<1){
      PrintStatus("Did not receive acknowledgement from client.");
      close(outsock);
      (void)alarm(timeremaining);
      hangingout=NO;
      RemoveClient(cycle_clients->ip_address,
		   cycle_clients->port);
      cycle_clients=old_client;
      dummy_client->next=message_clients;
      continue;
    } else {
      if (strcmp(buf,"client acknowledged")!=0){
	PrintStatus("Client did not properly acknowledge message.");
	close(outsock);
	(void)alarm(timeremaining);
	hangingout=NO;
	RemoveClient(cycle_clients->ip_address,
		     cycle_clients->port);
	cycle_clients=old_client;
	dummy_client->next=message_clients;
	continue;
      }
    }
    close(outsock);
    (void)alarm(timeremaining);
    hangingout=NO;
    old_client=cycle_clients;
  }
  free(dummy_client);
}

/*
  This routine puts a client onto the clients linked list for later messaging
*/
void RegisterClient(char *ipaddress,int port){
  registered_clients *newClient=NULL,*cycle_clients=NULL;
  int sock,tmplen,received=0,alreadyregistered=0;
  char tmp[BUFFSIZE];
  struct sockaddr_in mesgclient;
/*   struct timeval timeouts; */

  (void)snprintf(tmp,BUFFSIZE,"Attempting to contact client %s on port %d",ipaddress,port);
  PrintStatus(tmp);

  /* Create the TCP socket */
  if ((sock=socket(PF_INET,SOCK_STREAM,IPPROTO_TCP))<0){
    PrintStatus("Unable to create socket - this is a server error!");
    return;
  }
  socket_options(sock,SOCKETOPTIONS_OTHER);

  /* Construct the client sockaddr_in structure */
  memset(&mesgclient,0,sizeof(mesgclient));
  mesgclient.sin_family=AF_INET;
  mesgclient.sin_addr.s_addr=inet_addr(ipaddress);
  mesgclient.sin_port=htons(port);

  /* Establish connection */
  if (connect(sock,(struct sockaddr *)&mesgclient,sizeof(mesgclient))<0){
    close(sock);
    PrintStatus("Unable to connect to client!");
    return;
  }

  /* Send a test message to the server */
  (void)snprintf(tmp,BUFFSIZE,"Attempting to register this machine as a message client...");
  tmplen=(int)strlen(tmp);
  if (send(sock,tmp,tmplen,0)!=tmplen){
    PrintStatus("Could not send registration message to client.");
    close(sock);
    return;
  }

  /* Need to get the proper confirmation back from the client */
  /* should be "client acknowledged" */
  if ((received=recv(sock,tmp,BUFFSIZE-1,0))<1){
    PrintStatus("Did not receive acknowledgement from client.");
    close(sock);
    return;
  } else {
    if (strcmp(tmp,"client acknowledged")!=0){
      PrintStatus("Client did not properly acknowledge registration.");
      close(sock);
      return;
    }
  }

  /* If we get here the client can be registered */
  close(sock);
  /* check that it hasn't already been registered */
  for (cycle_clients=message_clients;cycle_clients!=NULL;cycle_clients=cycle_clients->next){
    if ((strcmp(cycle_clients->ip_address,ipaddress)==0)&&
	(cycle_clients->port==port)){
      /* this client has already been registered */
      /* don't register it again */
      alreadyregistered=1;
      break;
    }
  }
  if (alreadyregistered==0){ /* hasn't yet been registered */
    MALLOCV(newClient,sizeof(registered_clients));
    strcpy(newClient->ip_address,ipaddress);
    newClient->port=port;
    if (message_clients==NULL){ /* this is the first client to be registered */
      newClient->next=NULL;
    } else { /* add this client at the top of the list */
      newClient->next=message_clients;
    }
    message_clients=newClient;
  }
}

void RemoveClient(char *ip_address,int port_number){
  registered_clients *cycle_clients=NULL,*old_client=NULL;

  /* first we need to find the client that is to be removed */
  for (cycle_clients=message_clients;cycle_clients!=NULL;cycle_clients=cycle_clients->next){
    if ((strcmp(cycle_clients->ip_address,ip_address)==0)&&
	(cycle_clients->port==port_number)){
      if (old_client==NULL){ /* removing from the head of the list */
	message_clients=cycle_clients->next; /* point the head to the next node */
      } else { /* removing from somewhere other than the head */
	old_client->next=cycle_clients->next; /* skip over the removed node */
      }
      free(cycle_clients);
      return;
    }
    old_client=cycle_clients;
  }
}

/*
  This routine is called when the server starts, to initialise
  the values in the server_status structure
*/
void initialise_status(recorder *settings){
  status_server.status_recorder[0]='\0';                /* blank status */
  status_server.is_recording=NO;                        /* not recording */
  status_server.in_dir=NO;                              /* not in the output directory */
  status_server.recorder_pid=-1;                        /* no recorder process */
  status_server.status_checker[0]='\0';                 /* blank status */
  status_server.healthcheck_pid=-1;                     /* no health checker */
  status_server.current_settings=settings;              /* pointer to the current recording settings */
  MALLOCV(status_server.recording_settings,
	  sizeof(recorder));                            /* the settings from the last recorder instance */
  status_server.experiment_mode=EXPERIMENT_MANUAL;      /* manual recording */
  status_server.execute_experiment=AUTO_EXPERIMENT_YES; /* start experiments automatically */
  status_server.low_disk_action=CRITICAL_STOP;          /* stop recording when disk space runs out */
  status_server.recstatus=NULL;                         /* no recorder health information */
  status_server.rounded_start=ROUNDSTART_YES;           /* automatically choose next 10s boundary to
							   start recording */
  status_server.recording_to_disk=NULL;                 /* not recording */
  status_server.recording_path[0]='\0';                 /* not recording */
  status_server.interfaces=NULL;                        /* no network interfaces known yet */
  status_server.experiment_report=NULL;                 /* no experiment report file */
  status_server.remote_recorders=NULL;                  /* no remote recorders */
  status_server.target_list=NULL;                       /* no recording targets yet */
  message_clients=NULL;                                 /* no registered message clients */
  disk_one=NULL;                                        /* no recording devices */
  local_disks=NULL;
  remote_disks=NULL;
  vsib_errors=NULL;                                     /* no recorder error messages */
  hanging=NO;
  hangingout=NO;
}

/*
  This routine controls the starting of an experiment.
  It is called when the next experiment on the queue is
  due to begin, or when the user asks the experiment to begin.
*/
int StartExperiment(char *failure){
  int recval;
  time_t time_now;
  char timestring[BUFFSIZE],buf[BUFFSIZE];
  struct tm *temp_time;

  time(&time_now);
  PrintStatus("starting experiment routine");
  if (next_experiment->started==YES){
    /* the server may return here more than once between the recorder
       being started and the experiment actually starting, so we exit
       if we've already been asked to start this experiment */
    return(NO_ERROR);
  }
  
  if (time_now<next_experiment->start_time){
    /* we've got time before the experiment begins */
    /* this means the server has called this routine */
    /* automatically */

    /* first we check if the recorder is running */
    if (status_server.is_recording==YES){
      /* and stop it if it is */
      if ((recval=recorder_control(RECSTOP,status_server.current_settings,failure))!=NO_ERROR){
	return(recval);
      }
    }

  } else if ((time_now>next_experiment->start_time)&&(time_now<next_experiment->end_time)){
    /* we're being asked to start after the experiment */
    /* was supposed to start, most likely because the */
    /* user is controlling us */

    /* first we check if the recorder is running */
    if (status_server.is_recording==YES){
      /* and stop it if it is */
      if ((recval=recorder_control(RECSTOP,status_server.current_settings,failure))!=NO_ERROR){
	return(recval);
      }
    }

    /* need to change the start time to EXPSTARTBUF seconds from now */
    next_experiment->start_time=time_now+(time_t)EXPSTARTBUF;
    /* and then update the start time in the recording profile */
    temp_time=gmtime(&(next_experiment->start_time));
    (void)snprintf(next_experiment->record_settings->record_start_date,BUFFSIZE,"%4d%02d%02d",
		   (temp_time->tm_year)+1900,(temp_time->tm_mon)+1,temp_time->tm_mday);
    (void)snprintf(next_experiment->record_settings->record_start_time,BUFFSIZE,"%02d%02d%02d",
		   temp_time->tm_hour,temp_time->tm_min,temp_time->tm_sec);

  } else if (time_now>next_experiment->end_time){
    /* we've been asked to start after the end of the experiment */
    /* oops, something in the code has made an error */
    strcpy(failure,EXPERIMENT_OVER_ERR);
    return(EXPERIMENT_OVER);
  }

  /* make the current settings those of the experiment */
  copy_recorder(next_experiment->record_settings,status_server.current_settings);
/*   status_server.current_settings=next_experiment->record_settings; */

  /* figure out how long to record for */
  fuzzy_time((next_experiment->end_time-next_experiment->start_time),timestring);
  strcpy(status_server.current_settings->record_time,timestring);

  /* finally, we start the recorder */
  if ((recval=recorder_control(RECSTART,status_server.current_settings,failure))!=NO_ERROR){
    return(recval);
  }

  /* broadcast a message saying the experiment has started */
  thetimenow(timestring);
  next_experiment->started=YES;
  (void)snprintf(buf,BUFFSIZE,"Experiment %s started at %s.",next_experiment->experiment_id,timestring);
  BroadcastMessage(buf);

  /* the experiment has started properly */
  return(NO_ERROR);

}

int ExperimentReady(int action,char *failure){
  time_t time_now;
  int expdet=0,retval;
  experiment *old_experiment=NULL;

  time(&time_now);
  /* determine which experiment should be at the top of */
  /* the list */
  /* if the experiment is yet to start, or if there is more */
  /* than EXPSWITCHTIME seconds of recording time left, an experiment */
  /* stays at the top, otherwise we move to the next experiment */
  while((expdet==0)&&(next_experiment!=NULL)){
    if ((time_now<next_experiment->start_time)||
	((time_now>next_experiment->start_time)&&((next_experiment->end_time-time_now)>=(time_t)(2*STATUS_UPDATE+EXPUPDATEBLK)))){
      expdet=1;
    } else if ((next_experiment->end_time-time_now)<(time_t)(2*STATUS_UPDATE+EXPUPDATEBLK)) {
      old_experiment=next_experiment;
      next_experiment=next_experiment->next;
      free(old_experiment->record_settings); /* free our allocated memory */
      free(old_experiment);
    }
  }
  
  /* check that there is a valid experiment now */
  if (next_experiment==NULL){
    /* no more experiments */
    status_server.experiment_mode=EXPERIMENT_MANUAL; /* manual mode */
    strcpy(failure,EXP_QUEUE_DONE_ERR);
    return(EXP_QUEUE_DONE);
  }

  /* now we decide whether to start the experiment */
  /* if called automatically, we don't start vsib_record unless close to the */
  /* start time */
  if (action==READY_AUTO){
    if ((int)(next_experiment->start_time-time_now)>EXPSWITCHTIME){
      /* more than EXPSWITCHTIME seconds before the next experiment */
      /* we send back an error */
      status_server.experiment_mode=EXPERIMENT_MANUAL;
      strcpy(failure,EXP_START_LONG_ERR);
      return(EXP_START_LONG);
    }
  } /* if manually called, vsib_record for the next experiment is always started */
  if ((status_server.execute_experiment==AUTO_EXPERIMENT_YES)|| /* auto execute experiments */
      (status_server.execute_experiment==AUTO_EXPERIMENT_STOP)|| /* some experiment was stopped previously */
      (action==READY_MANUAL)) { /* manual command to start */
    /* start the experiment */
    if ((retval=StartExperiment(failure))!=NO_ERROR){
      return(retval);
    }
    /* we're now in experiment mode */
    status_server.experiment_mode=EXPERIMENT_QUEUE;
    /* switch back to auto execute if previous experiment was stopped */
    if (status_server.execute_experiment==AUTO_EXPERIMENT_STOP)
      status_server.execute_experiment=AUTO_EXPERIMENT_YES;
    /* return success code because we've started */
    return(NO_ERROR);
  } else {
    /* we can't start the experiment, but this is not really */
    /* an error, we've just progressed the experiment queue */
    /* nevertheless, we return an error code */
/*     status_server.experiment_mode=EXPERIMENT_MANUAL; /\* manual mode *\/ */
    status_server.experiment_mode=EXPERIMENT_QUEUE;
    strcpy(failure,EXP_START_WRONG_ERR);
    return(EXP_START_WRONG);
  }

}

void UpdateStatusSignal(int sig){
  char failure[BUFFSIZE],buf[BUFFSIZE],tmp[BUFFSIZE];
  int failcode,alarmtime;
  sigset_t block_signal;
  time_t time_now,start_time_diff,end_time_diff;
  remoterecorder *cyclerecorder=NULL;

  /* check which signal it is */
  sigemptyset(&block_signal);
  sigaddset(&block_signal,SIGINT); /* block Ctrl-C */
  if (sig==SIGALRM){
    /* we've been asked to auto-update */
    /* block the SIGCHLD signal until we're done */
    sigaddset(&block_signal,SIGCHLD);
    /* check whether we're hanging */
    if (hanging==YES){
      /* we need to close a hanging connection */
      close(connected_socket);
      hanging=NO;
    }
    if (hangingout==YES){
      /* we need to close a hanging connection */
      close(outsock);
      hangingout=NO;
    }
  } else if (sig==SIGCHLD){
    /* we've been called to clean up after a child has completed */
    /* block the SIGALRM signal until we're done */
/*     sigaddset(&block_signal,SIGALRM); */
    /* also, deal with one child at a time */
    sigaddset(&block_signal,SIGCHLD);
  }
  sigprocmask(SIG_BLOCK,&block_signal,NULL);

  /* the default time between alarms is STATUS_UPDATE seconds */
  alarmtime=STATUS_UPDATE;
  (void)signal(SIGALRM,UpdateStatusSignal);
  /* we give UpdateStatus 2 seconds to come back here */
  (void)alarm(2);

  if ((failcode=UpdateStatus(failure))!=NO_ERROR){
    (void)snprintf(buf,BUFFSIZE,"failed while updating status with error %d\n %s",failcode,failure);
    PrintStatus(buf);
  }

  /* check whether we should kill the receiver process */
  for (cyclerecorder=status_server.remote_recorders;cyclerecorder!=NULL;
       cyclerecorder=cyclerecorder->next){
    if (cyclerecorder->receiver!=NULL){
      if (time_now>=cyclerecorder->receiver->receiver_end){
	/* this process needs to die */
	kill(cyclerecorder->receiver->receiver_pid,SIGKILL);
      }
    }
  }

  /* check whether an experiment is close to being started or finished */
  time(&time_now);
  if (next_experiment!=NULL){
    start_time_diff=next_experiment->start_time-time_now;
    end_time_diff=next_experiment->end_time-time_now;
    if ((start_time_diff>=0)&&(start_time_diff<=(time_t)EXPSTARTBUF)){
      /* time to start the experiment */
      if ((failcode=ExperimentReady(READY_AUTO,failure))!=NO_ERROR){
	/* could be that the user doesn't want experiments to start automatically */
	if ((status_server.execute_experiment==AUTO_EXPERIMENT_YES)||
	    (status_server.execute_experiment==AUTO_EXPERIMENT_STOP)){
	  /* nope, something's gone wrong */
	  /* give a warning message */
	  (void)snprintf(buf,BUFFSIZE,"Unable to automatically start experiment %s.",next_experiment->experiment_id);
	  IssueWarning(buf);
	} else {
	  /* broadcast a message saying a new experiment is available */
	  (void)snprintf(buf,BUFFSIZE,"The next experiment in the queue (%s) is now available for manual start.",next_experiment->experiment_id);
	  BroadcastMessage(buf);
	}
      }
    } else if ((start_time_diff>=0)&&(start_time_diff<=(time_t)(STATUS_UPDATE+EXPUPDATEBLK))){
      /* the next experiment should start less than STATUS_UPDATE+EXPUPDATEBLK seconds from now */
      /* there will be no automatic UpdateStatus calls until the experiment starts since we need */
      /* the SIGALRM signal */
      alarmtime=(int)start_time_diff-EXPSTARTBUF;
      if (sig==SIGCHLD){
	/* we are here because a child has finished */
	/* need to set alarm ourselves */
	(void)signal(SIGALRM,UpdateStatusSignal);
	(void)alarm(alarmtime);
      } /* otherwise this will be done normally in this subroutine */
      (void)snprintf(buf,BUFFSIZE,"Experiment %s will begin in %d seconds.",next_experiment->experiment_id,alarmtime);
      PrintStatus(buf);
    } else if ((start_time_diff<0)&&(end_time_diff<=0)){
      /* time to advance to a new experiment */
      strcpy(tmp,next_experiment->experiment_id);
      if ((failcode=ExperimentReady(READY_AUTO,failure))!=NO_ERROR){
	/* probably just too much between now and the start of the next experiment */
	/* or that automatic experiment execution is turned off */
	if (failcode!=EXP_QUEUE_DONE){
	  /* there are more experiments in the queue */
	  (void)snprintf(buf,BUFFSIZE,"Experiment %s finished. Next experiment is now %s.",tmp,next_experiment->experiment_id);
	  /* come back almost immediately so we can start the next 
	     experiment if necessary */
	  alarmtime=(int)EXPSTARTBUF;
	  if (sig==SIGCHLD){
	    /* we are here because a child has finished */
	    /* need to set alarm ourselves */
	    (void)signal(SIGALRM,UpdateStatusSignal);
	    (void)alarm(alarmtime);
	  } /* otherwise this will be done normally in this subroutine */
	} else {
	  /* we've just finished the last experiment */
	  (void)snprintf(buf,BUFFSIZE,"Experiment %s finished. There are no more scheduled experiments.",tmp);
	}
	PrintStatus(buf);
      } /* we've actually already started the new experiment */
    } else if ((start_time_diff<0)&&(end_time_diff>0)&&(end_time_diff<=(time_t)(STATUS_UPDATE+EXPUPDATEBLK))){
      /* the current experiment will finish less than STATUS_UPDATE+EXPUPDATEBLK seconds from now */
      /* there will be no automatic UpdateStatus calls until the experiment finishes since we need */
      /* the SIGALRM signal */
      alarmtime=(int)end_time_diff;
      if (sig==SIGCHLD){
	/* we are here because a child has finished */
	/* need to set alarm ourselves */
	(void)signal(SIGALRM,UpdateStatusSignal);
	(void)alarm(alarmtime);
      } /* otherwise this will be done normally in this subroutine */
      (void)snprintf(buf,BUFFSIZE,"Experiment %s will finish in %d seconds.",next_experiment->experiment_id,alarmtime);
      PrintStatus(buf);
    } else {
      /* don't want to get here! */
/*       DebugStatus("I'm in a bad place!"); */
    }
  }	       
	       
  /* reset signals if need be */
  if (sig==SIGALRM){
    /* need to come back here every 20 seconds */
    (void)signal(SIGALRM,UpdateStatusSignal);
    (void)alarm(alarmtime);
  } else if (sig==SIGCHLD){
    /* still a child of ours running around (always do this anyway!) */
    (void)signal(SIGCHLD,UpdateStatusSignal);
  }
  
  /* unblock the signals */
  sigprocmask(SIG_UNBLOCK,&block_signal,NULL);
  
}

int UpdateStatus(char *failure){
  FILE *out_settings=NULL;
  experiment *cycle_experiment=NULL;
  char time_status[BUFFSIZE],time_string[BUFFSIZE],free_time[BUFFSIZE],free_status[BUFFSIZE];
  char tmp[BUFFSIZE],prevbuf[BUFFSIZE],cwd[BUFFSIZE];
  char logname[BUFFSIZE],tmpret[BUFFSIZE],temp_label[BUFFSIZE],recname[BUFFSIZE];
  char latest_quantities[BUFFSIZE],latest_statistics[BUFFSIZE];/* ,tmpfile[BUFFSIZE]; */
  time_t time_now,start_time,end_time;
  int year,month,day,hour,minute,second,duration,retval,datarate,freetime=0,status;
  int fin,itmp,disk_is_now_mounted,sysretval,getretval,foundreceiver;
  unsigned long long freespace,totalspace;
  outputdisk *firstdisk=NULL,*cycledisk=NULL;
  pid_t chldstatus;
  struct tm *temp_time;
  remoterecorder *cyclerecorders=NULL,dummyrecorder;

  PrintStatus("Updating status");

  /* check that the directory we're supposed to be in exists */
  if ((getcwd(cwd,BUFFSIZE))==NULL){
    /* the directory must be gone */
    sysretval=chdir(default_directory); /* change back to a path we assume exists */
    status_server.in_dir=NO; /* no longer in the directory */
  }

  /* check if the disks have been mounted/unmounted */
  for (cycledisk=disk_one;cycledisk!=NULL;cycledisk=cycledisk->next){
    disk_is_now_mounted=check_disk_mount(cycledisk->diskpath,cycledisk->filesystem);
    if ((disk_is_now_mounted==YES)&&(cycledisk->is_mounted==NO)){
      cycledisk->get_label=YES;
      status_server.update_remote_disks=YES;
    } else if ((disk_is_now_mounted==NO)&&(cycledisk->is_mounted==YES)){
      cycledisk->disklabel[0]='\0';
      status_server.update_remote_disks=YES;
    }
    cycledisk->is_mounted=disk_is_now_mounted;
  }

  for (cycledisk=disk_one;cycledisk!=NULL;cycledisk=cycledisk->next){
    if (cycledisk->get_label==YES){
      /* we need to look for a new label */
      strcpy(temp_label,cycledisk->diskpath);
      get_drive_serial(temp_label,cycledisk->filesystem,failure,DRIVES_LABEL);
      strcpy(cycledisk->disklabel,temp_label);
    }
  }
  
  /* if we have some remote hosts and they need updating, send them a list
     of our disks and their status */
  for (cycledisk=disk_one;cycledisk!=NULL;cycledisk=cycledisk->next){
    free_space(cycledisk->diskpath,&(cycledisk->freespace),
	       &(cycledisk->totalspace),failure);
    for (cyclerecorders=status_server.remote_recorders;cyclerecorders!=NULL;
	 cyclerecorders=cyclerecorders->next){
      if (cyclerecorders->evlbi_enabled==YES){
	/* we don't send disk information to an eVLBI recorder */
	continue;
      }
      /* prepare information */
      /* recorder name */
      conjugate_host(cyclerecorders->commonname,recname);
      if (strlen(cycledisk->disklabel)==0){
	strcpy(temp_label,"none");
      } else {
	strcpy(temp_label,cycledisk->disklabel);
	/* replace spaces with question marks, as when we read the label
	   as a remote recorder, we can't recognise spaces */
	while(strstr(temp_label," ")!=NULL){
	  (strstr(temp_label," "))[0]='?';
	}
      }
      /* now the string to send to the other recorder */
      if (status_server.update_remote_disks==YES){
	snprintf(tmp,BUFFSIZE,"<data>remote_disk=%s,%s,%llu,%llu,%s,%d,%d</data>",
		 recname,cycledisk->diskpath,cycledisk->freespace,cycledisk->totalspace,
		 temp_label,cycledisk->max_rate,cycledisk->is_mounted);
      } else {
	snprintf(tmp,BUFFSIZE,"<data>remote_disk=%s,%s,%llu</data>",recname,cycledisk->diskpath,
		 cycledisk->freespace);
      }
      getretval=PushRemote(tmp,cyclerecorders);
      if (getretval!=NO_ERROR){
	/* it's likely that this remote recorder is no longer viable, so we
	   remove it */
	dummyrecorder.next=cyclerecorders->next;
	RemoveRemote(cyclerecorders,REMOTEHOST_REMOVE_COMMSFAIL);
	cyclerecorders=&dummyrecorder;
      }
    }
  }
  status_server.update_remote_disks=NO;

  /* check if the day has changed and if we need to change the logfile */
  time(&time_now);
  temp_time=gmtime(&time_now);
  if ((temp_time->tm_yday+1)!=logday){
    /* day of the year has changed */
    fclose(logfile);
    logday=temp_time->tm_yday+1;
    (void)snprintf(logname,BUFFSIZE,"%s/recorder_log_%4d%03d",log_location,(temp_time->tm_year)+1900,logday);
    if ((logfile=fopen(logname,"a"))==NULL){
      PrintStatus("Cannot open new logfile for writing!");
      /* can't leave logfile as NULL or we'll get a segfault next write */
      logfile=fopen("/dev/null","a"); /* use the bottomless pit */
      logday=-1; /* try again next time */
    }
  }

  /* check that the start time/date is still in the future and reset it */
  /* if it isn't */
  if ((strlen(status_server.current_settings->record_start_date)!=0)&&
      (strlen(status_server.current_settings->record_start_time)!=0)){
    sscanf(status_server.current_settings->record_start_date,"%4d%2d%2d",
	   &year,&month,&day);
    sscanf(status_server.current_settings->record_start_time,"%2d%2d%2d",
	   &hour,&minute,&second);
    assign_time(year,month,day,hour,minute,second,&start_time);
    time(&time_now);
    if (time_now>start_time){
      /* specified start time/date is in the past */
      /* we only reset if we're not currently recording */
      if (status_server.is_recording==NO){
	data_handler("reset=record_start_time",tmpret,status_server.current_settings);
	data_handler("reset=record_start_date",tmpret,status_server.current_settings);
      }
    }
  }

  /* check running processes */
  /* this routine now also functions as the child handler */
  /* that is, it will be called on all SIGCHLD events to clean */
  /* up the child forks */
  
  /* recorder */
  if (status_server.recorder_pid>0){
    /* need to check if the recorder has exited */
    chldstatus=waitpid(status_server.recorder_pid,&status,WNOHANG);
    if ((chldstatus>0)|| /* waitpid says the process has completed */
	(chldstatus==-1)){ /* process doesn't exist, or not our child */
      /* if we've recovered after a crash, then the recorder won't be */
      /* our child, so we need to explicitly check for this */
      fin=1;
      if (chldstatus==-1){
	if ((itmp=checkrunning(status_server.recorder_pid,vsib_record_command))==1){
	  /* this is still the recorder */
	  /* we don't want to call the cleanup routine yet */
	  fin=0;
	}
      }
      if (fin==1){
	/* the recorder exited */
	/* check that the time is later than the expected end time */
	time(&time_now);
	if (time_now<status_server.recorder_end){
	  time_difference(time_now,status_server.recorder_end,time_string);
	  (void)snprintf(tmp,BUFFSIZE,"The recorder has stopped %s before its expected completion time",time_string);
	  IssueWarning(tmp);
	} else {
	  BroadcastMessage("The recorder has stopped as scheduled");
	}
	/* stop the health checker */
	if (health_control(HEALTH_STOP,failure)!=NO_ERROR)
	  PrintStatus(failure);
	/* we now copy the output from the recorder into the log file */
	(void)snprintf(tmp,BUFFSIZE,"%s/recorder_%d_output",tmp_directory,status_server.recorder_pid);
	/* 	PrintLog("================================================"); */
	/* 	PrintLog("Recorder finished... output of recorder follows."); */
	/* 	if ((recorder_output=fopen(tmp,"r"))!=NULL){ */
	/* 	  while((fgets(tmp,BUFFSIZE,recorder_output))!=NULL){ */
	/* 	    tmp[strlen(tmp)-1]='\0'; /\* remove the newline *\/ */
	/* 	    PrintLog(tmp); */
	/* 	    if ((sscanf(tmp,"at block = %d, opened file '%[^']'",&tmpblock,tmpfile))==2){ */
	/* 	      if (status_server.experiment_report!=NULL){ */
	/* 		fprintf(status_server.experiment_report,"%s\n",tmpfile); */
	/* 	      } */
	/* 	    } */
	/* 	  } */
	/* 	  fclose(recorder_output); */
	/* 	} else { */
	/* 	  PrintLog("no recorder output file found"); */
	/* 	} */
	/* 	PrintLog("================================================"); */
	status_server.recorder_pid=-1;
	status_server.is_recording=NO;
	directory_default(failure);
	status_server.in_dir=NO;
	/* close the experiment report file */
	if (status_server.experiment_report!=NULL){
	  fclose(status_server.experiment_report);
	  status_server.experiment_report=NULL;
	}
      }
    }
  }

  /* health checker */
  if (status_server.healthcheck_pid>0){
    chldstatus=waitpid(status_server.healthcheck_pid,&status,WNOHANG);
    if ((chldstatus>0)|| /* the child has finished */
	(chldstatus==-1)){ /* the process doesn't exist or is not our child */
      fin=1;
      if (chldstatus==-1){
	if ((checkrunning(status_server.healthcheck_pid,health_checker_command))==1){
	  /* this is still the health checker */
	  fin=0;
	}
      }
      if (fin==1){
	PrintStatus("Health checker finished.");
	status_server.healthcheck_pid=-1;
      }
    }
  } else if (status_server.is_recording==YES){
    /* the recorder is running, but the health checker isn't */
    /* we need to start one now */
    if (health_control(HEALTH_START,failure)!=NO_ERROR)
      PrintStatus(failure);
  }

  /* receiver */
  for (cyclerecorders=status_server.remote_recorders;cyclerecorders!=NULL;
       cyclerecorders=cyclerecorders->next){
    if (cyclerecorders->receiver!=NULL){
      /* need to check if the recorder has exited */
      chldstatus=waitpid(cyclerecorders->receiver->receiver_pid,&status,WNOHANG);
      if ((chldstatus>0)|| /* waitpid says the process has completed */
	  (chldstatus==-1)){ /* process doesn't exist, or not our child */
	/* if we've recovered after a crash, then the receiver won't be */
	/* our child, so we need to explicitly check for this */
	fin=1;
	if (chldstatus==-1){
	  if ((itmp=checkrunning(cyclerecorders->receiver->receiver_pid,recv_command))==1){
	    /* this is still the recorder */
	    /* we don't want to call the cleanup routine yet */
	    fin=0;
	  }
	}
	if (fin==1){
	  /* the recorder exited */
	  /* check that the time is later than the expected end time */
	  time(&time_now);
	  if (time_now<cyclerecorders->receiver->receiver_end){
	    time_difference(time_now,cyclerecorders->receiver->receiver_end,time_string);
	    (void)snprintf(tmp,BUFFSIZE,"The receiver has stopped %s before its expected completion time",time_string);
	    IssueWarning(tmp);
	  } else {
	    BroadcastMessage("The receiver has stopped as scheduled");
	  }
	  /* we now copy the output from the receiver into the log file */
	  (void)snprintf(tmp,BUFFSIZE,"%s/receiver_%d_output",tmp_directory,status_server.receiver_pid);
	  directory_default(failure);
	  status_server.in_dir=NO;
	  FREE(cyclerecorders->receiver);
	}
      }
    }
  }

  /* write out the status to file, so we can recover server crashes */
  (void)snprintf(tmp,BUFFSIZE,"%s/recorder_server_status",tmp_directory);
  if ((out_settings=fopen(tmp,"w"))==NULL){
    strcpy(failure,STATUS_WRITE_FAIL_ERR);
    return(STATUS_WRITE_FAIL);
  }
  fprintf(out_settings,"%d %d %ld %d\n",status_server.is_recording,status_server.recorder_pid,status_server.recorder_start,status_server.healthcheck_pid);
  fprintf(out_settings,"\n");
  fprintf(out_settings,"%d %d\n",status_server.experiment_mode,status_server.execute_experiment);
  if (next_experiment!=NULL){
    /* output the name of the experiments so they can be loaded in again */
    for (cycle_experiment=next_experiment;cycle_experiment!=NULL;cycle_experiment=cycle_experiment->next){
      fprintf(out_settings,"%s\n",cycle_experiment->experiment_id);
    }
  }
  fclose(out_settings);

  /* get the current time */
  time(&time_now);

  /* generate status messages */

  /* recorder status */
  if (status_server.is_recording==YES){
    /* work out the time until completion */
    if (strlen(status_server.recording_settings->record_start_time)>0){ /* the recording may not have started */
      sscanf(status_server.recording_settings->record_start_date,"%4d%2d%2d",&year,&month,&day);
      sscanf(status_server.recording_settings->record_start_time,"%2d%2d%2d",&hour,&minute,&second);
      assign_time(year,month,day,hour,minute,second,&start_time);
      duration=timeinseconds(status_server.recording_settings->record_time);
      end_time=start_time+duration;
      if (time_now<start_time){
	time_difference(time_now,start_time,time_string);
	(void)snprintf(time_status,BUFFSIZE,"Recording will begin in %s.",time_string);
      } else if (time_now<end_time){
	time_difference(time_now,end_time,time_string);
	(void)snprintf(time_status,BUFFSIZE,"Recording will finish in %s.",time_string);
      }
    } else { /* the recording has started */
      duration=timeinseconds(status_server.recording_settings->record_time);
      end_time=status_server.recorder_start+duration;
      time_difference(time_now,end_time,time_string);
      (void)snprintf(time_status,BUFFSIZE,"Recording will finish in %s.",time_string);
    }

    /* determine the amount of free space on the recording drive */
    if (status_server.recording_settings->targetrecorder==NULL){
      retval=free_space(status_server.recording_settings->recordingdisk->diskpath,
			&freespace,&totalspace,failure);
    } else {
      if (status_server.recording_settings->recordingdisk!=NULL){
	freespace=status_server.recording_settings->recordingdisk->freespace;
	totalspace=status_server.recording_settings->recordingdisk->totalspace;
      } else {
	/* probably have an eVLBI recorder */
	freespace=0;
	totalspace=0;
      }
    }
    /* determine the recording data rate */
    datarate=recordingdatarate(status_server.recording_settings);
    /* calculate the available recording time */
    if ((status_server.recording_settings->targetrecorder==NULL)||
	((status_server.recording_settings->targetrecorder!=NULL)&&
	 (status_server.recording_settings->targetrecorder->evlbi_enabled==NO))){
      freetime=recordingfreetime(freespace,datarate);
      /* is the disk space getting low? */
      if (freetime<=DISK_WARNING1){
	if (freetime<=DISK_WARNING2){
	  if (freetime<=DISK_CRITICAL){
	    (void)snprintf(tmp,BUFFSIZE,"DISK SPACE CRITICAL! %d seconds remaining on %s"
			   ,freetime,status_server.recording_settings->recordingdisk->diskpath);
	    if (status_server.disk_warning_level!=WARNING_CRITICAL){
	      IssueWarning(tmp);
	      status_server.disk_warning_level=WARNING_CRITICAL;
	    }
	    strcpy(tmp,"DISK SPACE CRITICAL!\n");
	    /* do something at this point */
	    strcpy(prevbuf,tmp);
	    if ((retval=disk_critical_action(tmpret))!=NO_ERROR){
	      (void)snprintf(tmp,BUFFSIZE,"%sCould not execute critical action: %s\n",prevbuf,tmpret);
	    } else {
	      (void)snprintf(tmp,BUFFSIZE,"%sAction executed: %s\n",prevbuf,tmpret);
	    }
	  } else {
	    (void)snprintf(tmp,BUFFSIZE,"DISK SPACE LOW - SECOND WARNING: %d seconds remaining on %s"
			   ,freetime,status_server.recording_settings->recordingdisk->diskpath);
	    if (status_server.disk_warning_level!=WARNING_TWO){
	      IssueWarning(tmp);
	      status_server.disk_warning_level=WARNING_TWO;
	    }
	    strcpy(tmp,"DISK SPACE LOW - SECOND WARNING\n");
	  }
	} else {
	  (void)snprintf(tmp,BUFFSIZE,"DISK SPACE LOW - FIRST WARNING: %d seconds remaining on %s"
			 ,freetime,status_server.recording_settings->recordingdisk->diskpath);
	  if (status_server.disk_warning_level!=WARNING_ONE){
	    IssueWarning(tmp);
	    status_server.disk_warning_level=WARNING_ONE;
	  }
	  strcpy(tmp,"DISK SPACE LOW - FIRST WARNING\n");
	}
      } else {
	tmp[0]='\0';
      }
      time_difference((time_t)0,(time_t)freetime,free_time); /* as a string */
      if (status_server.recording_settings->targetrecorder==NULL){
	(void)snprintf(free_status,BUFFSIZE,"%sRemaining recording time on local:%s: %s",
		       tmp,status_server.recording_settings->recordingdisk->diskpath,free_time);
      } else {
	(void)snprintf(free_status,BUFFSIZE,"%sRemaining recording time on %s:%s: %s",
		       tmp,status_server.recording_settings->targetrecorder->commonname,
		       status_server.recording_settings->recordingdisk->diskpath,free_time);
      }
    }
      
    /* check for errors */
    retval=RecorderErrors(latest_quantities);

    if (status_server.recstatus!=NULL){
      strcpy(latest_statistics,status_server.recstatus->statistics);
      strcpy(status_server.lastrec.statistics,status_server.recstatus->statistics);
    } else {
      strcpy(latest_statistics,"No sampler statistics available.");
    }    

    /* make the recorder status message */
    (void)snprintf(status_server.status_recorder,BUFFSIZE,
		   "Recorder process running.\n%s\n%s\n%s\n%s\n%s",
		   status_server.recorder_command,time_status,free_status,latest_quantities,
		   latest_statistics);
  } else {
    (void)snprintf(status_server.status_recorder,BUFFSIZE,"Recorder process not running.");
    retval=RecorderErrors(latest_quantities);
    if (strlen(latest_quantities)>0){
      strcpy(tmp,status_server.status_recorder);
      (void)snprintf(status_server.status_recorder,BUFFSIZE,"%s\nLast recording:\n%s%s",
		     tmp,latest_quantities,status_server.lastrec.statistics);
    }
  }

  /* receiver status */
  status_server.status_receiver[0]='\0';
  foundreceiver=0;
  for (cyclerecorders=status_server.remote_recorders;
       cyclerecorders!=NULL;cyclerecorders=cyclerecorders->next){
    if (cyclerecorders->receiver!=NULL){
      (void)snprintf(tmp,BUFFSIZE,
		     "Receiver process running.\n%s\n",status_server.receiver_command);
      strcat(status_server.status_receiver,tmp);
      /* determine the amount of free space on the recording drive */
      retval=free_space(cyclerecorders->receiver->recordingdisk->diskpath,
			&freespace,&totalspace,failure);
      /* determine the recording data rate */
      datarate=recordingdatarate(status_server.current_settings);
      /* calculate the available recording time */
      freetime=recordingfreetime(freespace,datarate);
      time_difference((time_t)0,(time_t)freetime,free_time); /* as a string */
      (void)snprintf(tmp,BUFFSIZE,"Receiving to directory %s, free space %s",
		     cyclerecorders->receiver->recordingdisk->diskpath,free_time);
      strcat(status_server.status_receiver,tmp);
      foundreceiver++;
    }
  } 
  if (foundreceiver==0){
    (void)snprintf(status_server.status_receiver,BUFFSIZE,
		   "Receiver not running.\n");
  }

  /* server/system status */
  /* status message looks like: */
  /* Current Time: HH:MM:SS dd/mm/yyyy UTC */
  thetimenow(tmp);
  (void)snprintf(status_server.status_server,BUFFSIZE,"Current Time: %s",tmp);
  /* Server uptime: DDd HH:MM */
  time_difference(status_server.server_start,time_now,tmp);
  strcpy(prevbuf,status_server.status_server);
  (void)snprintf(status_server.status_server,BUFFSIZE,"%s\nServer uptime: %s",prevbuf,tmp);
  /* Current Experiment: ----- */
  if (status_server.experiment_mode==EXPERIMENT_QUEUE){
    if (next_experiment->start_time>time_now){ /* no experiment currently scheduled */
      (void)snprintf(tmp,BUFFSIZE,"Current Experiment: none");
    } else {
      (void)snprintf(tmp,BUFFSIZE,"Current Experiment: %s",next_experiment->experiment_id);
    }
  } else {
    if (strcmp(status_server.current_settings->directory_name,"FringeCheck/")==0)
      (void)snprintf(tmp,BUFFSIZE,"Current Experiment: %s (manual)",
		     status_server.current_settings->filename_prefix);
    else
      (void)snprintf(tmp,BUFFSIZE,"Current Experiment: %s (manual)",
		     status_server.current_settings->directory_name);
  }
  strcpy(prevbuf,status_server.status_server);
  (void)snprintf(status_server.status_server,BUFFSIZE,"%s\n%s",prevbuf,tmp);
  /* Time until end: ------- */
  if (status_server.experiment_mode==EXPERIMENT_QUEUE){
    if (next_experiment->start_time>time_now){
      (void)snprintf(tmp,BUFFSIZE,"Time until end: N/A");
    } else {
      time_difference(time_now,next_experiment->end_time,time_string);
      (void)snprintf(tmp,BUFFSIZE,"Time until end: %s",time_string);
    }
  } else {
    if (status_server.is_recording==YES){
      end_time=status_server.recorder_end;
      time_difference(time_now,end_time,time_string);
      (void)snprintf(tmp,BUFFSIZE,"Time until end: %s",time_string);
    } else {
      if (strlen(status_server.current_settings->record_start_time)>0){
	sscanf(status_server.current_settings->record_start_date,"%4d%2d%2d",
	       &year,&month,&day);
	sscanf(status_server.current_settings->record_start_time,"%2d%2d%2d",
	       &hour,&minute,&second);
	assign_time(year,month,day,hour,minute,second,&start_time);
	end_time=timeinseconds(status_server.current_settings->record_time);
	end_time+=start_time;
	time_difference(time_now,end_time,time_string);
	if (time_now<start_time){
	  (void)snprintf(tmp,BUFFSIZE,"Time until end: %s (not started)",time_string);
	} else if (time_now<end_time){
	  (void)snprintf(tmp,BUFFSIZE,"Time until end: %s (start passed)",time_string);
	}
      } else
	(void)snprintf(tmp,BUFFSIZE,"Time until end: not known");
    }
  }    
  strcpy(prevbuf,status_server.status_server);
  (void)snprintf(status_server.status_server,BUFFSIZE,"%s\n%s",prevbuf,tmp);
  /* Next Experiment: ------- */
  if (next_experiment!=NULL){
    if (next_experiment->start_time>time_now){
      (void)snprintf(tmp,BUFFSIZE,"Next Experiment: %s",next_experiment->experiment_id);
    } else {
      if (next_experiment->next!=NULL){
	(void)snprintf(tmp,BUFFSIZE,"Next Experiment: %s",next_experiment->next->experiment_id);
      } else {
	(void)snprintf(tmp,BUFFSIZE,"Next Experiment: N/A");
      }
    }
  } else {
    (void)snprintf(tmp,BUFFSIZE,"Next Experiment: N/A");
  }
  strcpy(prevbuf,status_server.status_server);
  (void)snprintf(status_server.status_server,BUFFSIZE,"%s\n%s",prevbuf,tmp);
  /* Time until start: ------- */
  if (next_experiment!=NULL){
    if (next_experiment->start_time>time_now){
      time_difference(time_now,next_experiment->start_time,time_string);
      (void)snprintf(tmp,BUFFSIZE,"Time until start: %s",time_string);
    } else {
      if (next_experiment->next!=NULL){
	time_difference(time_now,next_experiment->next->start_time,time_string);
	(void)snprintf(tmp,BUFFSIZE,"Time until start: %s",time_string);
      } else {
	(void)snprintf(tmp,BUFFSIZE,"Time until start: N/A");
      }
    }
  } else {
    (void)snprintf(tmp,BUFFSIZE,"Time until start: N/A");
  }
  strcpy(prevbuf,status_server.status_server);
  (void)snprintf(status_server.status_server,BUFFSIZE,"%s\n%s",prevbuf,tmp);
  /* Current settings: ------- */
  if (status_server.current_settings->fringe_test){
    strcpy(tmp,"Y");
  } else {
    strcpy(tmp,"N");
  }
  strcpy(prevbuf,status_server.status_server);
  (void)snprintf(status_server.status_server,BUFFSIZE,"%s\nCurrent settings: CH=%s BW=%d FT=%s BR=%d",
	  prevbuf,status_server.current_settings->compression,status_server.current_settings->bandwidth,
	  tmp,recordingdatarate(status_server.current_settings));
  /* Recording Status: ------- */
  if (status_server.is_recording==YES){
    (void)snprintf(tmp,BUFFSIZE,"Recording Status: Recording.");
    strcpy(prevbuf,status_server.status_server);
    (void)snprintf(status_server.status_server,BUFFSIZE,"%s\n%s",prevbuf,tmp);
    if (status_server.recording_settings->fringe_test){
      strcpy(tmp,"Y");
    } else {
      strcpy(tmp,"N");
    }
    strcpy(prevbuf,status_server.status_server);
    (void)snprintf(status_server.status_server,BUFFSIZE,"%s\nCurrent recording settings: CH=%s BW=%d FT=%s BR=%d",
		   prevbuf,status_server.recording_settings->compression,
		   status_server.recording_settings->bandwidth,tmp,
		   recordingdatarate(status_server.recording_settings));
		   
  } else {
    (void)snprintf(tmp,BUFFSIZE,"Recording Status: Not recording.");
    strcpy(prevbuf,status_server.status_server);
    (void)snprintf(status_server.status_server,BUFFSIZE,"%s\n%s",prevbuf,tmp);
  }
  /* Receiving Status: -------- */
  if (foundreceiver>0)
    (void)snprintf(tmp,BUFFSIZE,"Receiving Status: Receiving (%d receivers).",
		   foundreceiver);
  else
    (void)snprintf(tmp,BUFFSIZE,"Receiving Status: Not receiving.");
  strcpy(prevbuf,status_server.status_server);
  (void)snprintf(status_server.status_server,BUFFSIZE,"%s\n%s",prevbuf,tmp);
  /* Disk Space: */
  strcpy(prevbuf,status_server.status_server);
  (void)snprintf(status_server.status_server,BUFFSIZE,"%s\nDisk Space:",prevbuf);
  /*   disk n: -------- */
  firstdisk=disk_one;
  for (cycledisk=firstdisk;cycledisk!=NULL;cycledisk=cycledisk->next){
    if (cycledisk->is_mounted==YES){
      free_space(cycledisk->diskpath,&freespace,&totalspace,failure);
      cycledisk->freespace=freespace;
      cycledisk->totalspace=totalspace;
      time_difference((time_t)0,
		      (time_t)recordingfreetime(freespace,
						recordingdatarate(status_server.current_settings)),
		      time_string);
      (void)snprintf(tmp,BUFFSIZE,"  disk %d [%s \"%s\"]: TOTAL: %d MB FREE: %d MB (%d%%) TIME: %s",
		     cycledisk->disknumber,cycledisk->diskpath,cycledisk->disklabel,
		     (int)(totalspace/1024.0),(int)(freespace/1024.0),
		     (int)(((float)freespace/(float)totalspace)*100.0),time_string);
      strcpy(prevbuf,status_server.status_server);
      (void)snprintf(status_server.status_server,BUFFSIZE,"%s\n%s",prevbuf,tmp);
    }
  }
  /* now go through the remote disks that we know about */
  for (cyclerecorders=status_server.remote_recorders;cyclerecorders!=NULL;
       cyclerecorders=cyclerecorders->next){
    for (cycledisk=cyclerecorders->available_disks;cycledisk!=NULL;
	 cycledisk=cycledisk->next){
      time_difference((time_t)0,
		      (time_t)recordingfreetime(cycledisk->freespace,
						recordingdatarate(status_server.current_settings)),
		      time_string);
      (void)snprintf(tmp,BUFFSIZE,"  disk %d [%s:%s \"%s\"]: TOTAL: %d MB FREE: %d MB (%d%%) TIME: %s",
		     cycledisk->disknumber,cyclerecorders->commonname,cycledisk->diskpath,
		     cycledisk->disklabel,(int)(cycledisk->totalspace/1024.0),
		     (int)(cycledisk->freespace/1024.0),
		     (int)(((float)cycledisk->freespace/(float)cycledisk->totalspace)*100.0),
		     time_string);
      strcpy(prevbuf,status_server.status_server);
      (void)snprintf(status_server.status_server,BUFFSIZE,"%s\n%s",prevbuf,tmp);
    }
  }
  /* finally we print out info about eVLBI targets */
  for (cyclerecorders=status_server.remote_recorders;cyclerecorders!=NULL;
       cyclerecorders=cyclerecorders->next){
    if (cyclerecorders->evlbi_enabled==YES){
      (void)snprintf(tmp,BUFFSIZE,"  disk [%s:]: eVLBI recorder",cyclerecorders->commonname);
      strcpy(prevbuf,status_server.status_server);
      (void)snprintf(status_server.status_server,BUFFSIZE,"%s\n%s",prevbuf,tmp);
    }
  }
  /* first we check that we know about the network interfaces */
  interface_config(failure);

  /* now we update the statistics */
  if_readlist_proc();

  return(NO_ERROR);

}

void if_readlist_proc(void){
  FILE *fh=NULL;
  char buf[BUFFSIZE],name[IFNAMSIZ],*s=NULL,*tt=NULL;
  int procnetdev_vsn;
  network_interfaces *cycle_interfaces=NULL;

  /* mark statistics as invalid for now */
  for (cycle_interfaces=status_server.interfaces;
       cycle_interfaces!=NULL;cycle_interfaces=cycle_interfaces->next){
    cycle_interfaces->statistics_valid=NO;
  }

  /* open the kernel interfaces file */
  fh=fopen("/proc/net/dev","r");
  if (!fh)
    return;

  tt=fgets(buf,BUFFSIZE,fh); /* eat line */
  tt=fgets(buf,BUFFSIZE,fh);

  procnetdev_vsn=procnetdev_version(buf);

  while (fgets(buf,BUFFSIZE,fh)){
    s=get_name(name,buf);
    for (cycle_interfaces=status_server.interfaces;
	 cycle_interfaces!=NULL;cycle_interfaces=cycle_interfaces->next){
      if (strcmp(name,cycle_interfaces->interface_name)==0){
	/* we've found the right interface */
	get_dev_fields(s,cycle_interfaces,procnetdev_vsn);
	/* mark the statistics as valid */
	cycle_interfaces->statistics_valid=YES;
	break;
      }
    }
  }
  fclose(fh);

  return;
}

int get_dev_fields(char *bp,network_interfaces *ife,int procnetdev_vsn)
{
  switch (procnetdev_vsn) {
  case 3:
    sscanf(bp,
	   "%Lu %Lu %lu %lu %lu %lu %lu %lu %Lu %Lu %lu %lu %lu %lu %lu %lu",
	   &ife->stats.rx_bytes,
	   &ife->stats.rx_packets,
	   &ife->stats.rx_errors,
	   &ife->stats.rx_dropped,
	   &ife->stats.rx_fifo_errors,
	   &ife->stats.rx_frame_errors,
	   &ife->stats.rx_compressed,
	   &ife->stats.rx_multicast,
	   
	   &ife->stats.tx_bytes,
	   &ife->stats.tx_packets,
	   &ife->stats.tx_errors,
	   &ife->stats.tx_dropped,
	   &ife->stats.tx_fifo_errors,
	   &ife->stats.collisions,
	   &ife->stats.tx_carrier_errors,
	   &ife->stats.tx_compressed);
    break;
  case 2:
    sscanf(bp, "%Lu %Lu %lu %lu %lu %lu %Lu %Lu %lu %lu %lu %lu %lu",
	   &ife->stats.rx_bytes,
	   &ife->stats.rx_packets,
	   &ife->stats.rx_errors,
	   &ife->stats.rx_dropped,
	   &ife->stats.rx_fifo_errors,
	   &ife->stats.rx_frame_errors,
	   
	   &ife->stats.tx_bytes,
	   &ife->stats.tx_packets,
	   &ife->stats.tx_errors,
	   &ife->stats.tx_dropped,
	   &ife->stats.tx_fifo_errors,
	   &ife->stats.collisions,
	   &ife->stats.tx_carrier_errors);
    ife->stats.rx_multicast = 0;
    break;
  case 1:
    sscanf(bp, "%Lu %lu %lu %lu %lu %Lu %lu %lu %lu %lu %lu",
	   &ife->stats.rx_packets,
	   &ife->stats.rx_errors,
	   &ife->stats.rx_dropped,
	   &ife->stats.rx_fifo_errors,
	   &ife->stats.rx_frame_errors,
	   
	   &ife->stats.tx_packets,
	   &ife->stats.tx_errors,
	   &ife->stats.tx_dropped,
	   &ife->stats.tx_fifo_errors,
	   &ife->stats.collisions,
	   &ife->stats.tx_carrier_errors);
    ife->stats.rx_bytes = 0;
    ife->stats.tx_bytes = 0;
    ife->stats.rx_multicast = 0;
    break;
  }
  return 0;
}

char *get_name(char *name, char *p)
{
  while (isspace(*p))
    p++;
  while (*p) {
    if (isspace(*p))
      break;
    if (*p == ':') {	/* could be an alias */
      char *dot = p++;
      while (*p && isdigit(*p)) p++;
      if (*p == ':') {
	/* Yes it is, backup and copy it. */
	p = dot;
	*name++ = *p++;
	while (*p && isdigit(*p)) {
	  *name++ = *p++;
	}
      } else {
	/* No, it isn't */
	p = dot;
      }
      p++;
      break;
    }
    *name++ = *p++;
  }
  *name++ = '\0';
  return p;
}

int procnetdev_version(char *buf)
{
    if (strstr(buf, "compressed"))
	return 3;
    if (strstr(buf, "bytes"))
	return 2;
    return 1;
}

int RecorderErrors(char *returned_lines){
  recorder_health *cycle_health=NULL;
  int total_file_time=0,spacings=0,time_diff=0,retval=-1;
  int PPS_total=0,PPS_skips=0,max_file_time=0;
  float av_file_time=0.0;
  char line1[BUFFSIZE],line2[BUFFSIZE],line3[BUFFSIZE],line4[BUFFSIZE];
  char line5[BUFFSIZE],line6[BUFFSIZE];
  struct tm last_file_time;

  if (status_server.is_recording==YES){
    /* check how many PPS signals have been missed so far and */
    /* how long the time between files is */
    for (cycle_health=status_server.recstatus;
	 cycle_health!=NULL;cycle_health=cycle_health->next){
      PPS_total++;
      if (cycle_health->PPS_OK==NO)
	PPS_skips++;
      if (cycle_health->next!=NULL){
	time_diff=(int)((cycle_health->file_time)-((cycle_health->next)->file_time));
	if (time_diff>max_file_time)
	  max_file_time=time_diff;
	total_file_time+=time_diff;
	spacings++;
      }
    }
    
    av_file_time=(float)total_file_time/(float)spacings;
    
    /* default status lines if recording status not sent yet */
    snprintf(line1,BUFFSIZE,"(unknown)");
    snprintf(line2,BUFFSIZE,"(unknown)");
    snprintf(line3,BUFFSIZE,"(unknown)");
    snprintf(line4,BUFFSIZE,"(unknown)");
    snprintf(line5,BUFFSIZE,"(unknown)");
    snprintf(line6,BUFFSIZE,"(unknown)");
  }

  /* check the latest file to see if the recorder is currently OK */
  if ((status_server.recstatus!=NULL)||
      ((status_server.is_recording==NO)&&
       (status_server.lastrec.valid_info==YES))){
    if (status_server.is_recording==YES){
      (void)gmtime_r(&(status_server.recstatus->file_time),&last_file_time);
      (void)snprintf(line1,BUFFSIZE,"last file opened at day %d, %02d:%02d:%02d, block %d (%s)",
		     last_file_time.tm_yday+1,last_file_time.tm_hour,
		     last_file_time.tm_min,last_file_time.tm_sec,
		     status_server.recstatus->file_block,
		     status_server.recstatus->file_name);
      status_server.lastrec.file_time=status_server.recstatus->file_time;
      status_server.lastrec.file_block=status_server.recstatus->file_block;
      strcpy(status_server.lastrec.file_name,
	     status_server.recstatus->file_name);
    } else {
      (void)gmtime_r(&(status_server.lastrec.file_time),&last_file_time);
      (void)snprintf(line1,BUFFSIZE,"last file opened at day %d, %02d:%02d:%02d, block %d (%s)",
		     last_file_time.tm_yday+1,last_file_time.tm_hour,
		     last_file_time.tm_min,last_file_time.tm_sec,
		     status_server.lastrec.file_block,
		     status_server.lastrec.file_name);
    }
    if (status_server.is_recording==YES){
      if (status_server.recording_settings->targetrecorder==NULL){
	(void)snprintf(line6,BUFFSIZE,"recording to disk local:%s (%s), path %s",
		       status_server.recording_settings->recordingdisk->diskpath,
		       status_server.recording_settings->recordingdisk->disklabel,
		       status_server.recording_settings->directory_name);
      } else if (status_server.recording_settings->targetrecorder->evlbi_enabled==NO){
	(void)snprintf(line6,BUFFSIZE,"recording to disk %s:%s (%s), path %s",
		       status_server.recording_settings->targetrecorder->commonname,
		       status_server.recording_settings->recordingdisk->diskpath,
		       status_server.recording_settings->recordingdisk->disklabel,
		       status_server.recording_settings->directory_name);
      } else {
	(void)snprintf(line6,BUFFSIZE,"recording to eVLBI machine %s",
		       status_server.recording_settings->targetrecorder->commonname);
      }
      status_server.lastrec.recorded_to_disk=
	status_server.recording_settings->recordingdisk;
      strcpy(status_server.lastrec.recorded_path,
	     status_server.recording_settings->directory_name);
    } else {
      if (status_server.recording_settings->targetrecorder==NULL){
	(void)snprintf(line6,BUFFSIZE,"recording to disk local:%s (%s), path %s",
		       status_server.lastrec.recorded_to_disk->diskpath,
		       status_server.lastrec.recorded_to_disk->disklabel,
		       status_server.lastrec.recorded_path);
      } else {
	if (status_server.lastrec.recorded_to_disk!=NULL){
	  (void)snprintf(line6,BUFFSIZE,"recording to disk %s:%s (%s), path %s",
			 status_server.recording_settings->targetrecorder->commonname,
			 status_server.lastrec.recorded_to_disk->diskpath,
			 status_server.lastrec.recorded_to_disk->disklabel,
			 status_server.lastrec.recorded_path);
	} else {
	  (void)snprintf(line6,BUFFSIZE,"recording to disk %s:",
			 status_server.recording_settings->targetrecorder->commonname);
	}
      }
    }
    if (status_server.is_recording==YES){
      (void)snprintf(line2,BUFFSIZE,"BIGBUF free memory = %d MB (%d%%)",
		     status_server.recstatus->BIGBUF_level,
		     status_server.recstatus->BIGBUF_pct);
      status_server.lastrec.BIGBUF_level=status_server.recstatus->BIGBUF_level;
      status_server.lastrec.BIGBUF_pct=status_server.recstatus->BIGBUF_pct;
    } else {
      (void)snprintf(line2,BUFFSIZE,"BIGBUF free memory = %d MB (%d%%)",
		     status_server.lastrec.BIGBUF_level,
		     status_server.lastrec.BIGBUF_pct);
    }
    if (status_server.is_recording==YES){
      strcpy(line3,status_server.recstatus->PPS_message);
      strcpy(status_server.lastrec.PPS_message,status_server.recstatus->PPS_message);
    } else {
      strcpy(line3,status_server.lastrec.PPS_message);
    }
    if (status_server.is_recording==YES){
      (void)snprintf(line4,BUFFSIZE,"PPS missed %d times from last %d files",
		     PPS_skips,PPS_total);
      status_server.lastrec.total_files=PPS_total;
      status_server.lastrec.total_skips=PPS_skips;
    } else {
      (void)snprintf(line4,BUFFSIZE,"PPS missed %d times from last %d files",
		     status_server.lastrec.total_skips,
		     status_server.lastrec.total_files);
    }
    if (status_server.is_recording==YES){
      (void)snprintf(line5,BUFFSIZE,"Time between files: avg = %6.2f s, max = %d s",
		     av_file_time,max_file_time);
      status_server.lastrec.valid_info=YES;
    } else
      line5[0]='\0';
    if (status_server.is_recording==YES){
      if (status_server.recstatus->PPS_OK==YES){
	retval=RECORDER_OK;
      } else {
	retval=RECORDER_ERROR;
      }
      status_server.lastrec.PPS_OK=status_server.recstatus->PPS_OK;
    } else
      retval=RECORDER_OK;
  }
  if ((status_server.is_recording==YES)||
      ((status_server.is_recording==NO)&&
       (status_server.lastrec.valid_info==YES)))
    (void)snprintf(returned_lines,BUFFSIZE,"%s\n%s\n%s\n%s\n%s\n%s",
		   line1,line6,line2,line3,line4,line5);
  else
    returned_lines[0]='\0';
  return(retval);
      
}

/*
  This routine takes a time expressed as
  nh (hours) or
  nm (minutes) or
  ns (seconds)
  and converts it into an integer value representing
  the number of seconds in that time
*/
int timeinseconds(char *time_expression){
  int time_i=-1,temp_i;
  float temp_f;
  char buf[BUFFSIZE];
  
  /* is it a float or int time? */
/*   (void)snprintf(buf,BUFFSIZE,"converting %s to seconds",time_expression); */
/*   DebugStatus(buf); */
  if ((strstr(time_expression,"."))==NULL){ /* no decimal point found -> integer time */
    if ((sscanf(time_expression,"%d%1s",&temp_i,buf))==2){
      if (buf[0]=='h'){
	time_i=temp_i*3600;
      } else if (buf[0]=='m'){
	time_i=temp_i*60;
      } else if (buf[0]=='s'){
	time_i=temp_i;
      }
    }
  } else { /* a decimal point was found -> float time */
    if ((sscanf(time_expression,"%f%1s",&temp_f,buf))==2){
      if (buf[0]=='h'){
	time_i=(int)(temp_f*3600.0); /* conversion to int will probably result in an error of up to 1 second */
      } else if (buf[0]=='m'){
	time_i=(int)(temp_f*60.0);
      } else if (buf[0]=='s'){
	time_i=(int)temp_f;
      }
    }
  }

/*   (void)snprintf(buf,BUFFSIZE,"result is %d seconds",time_i); */
/*   DebugStatus(buf); */
  return (time_i);
}

/*
  This routine takes a time and turns it into a date string as
  HH:MM:SS dd/mm/yyyy
*/
void timeasstring(time_t that_time,char *time_string){
  struct tm *time_that;

  time_that=gmtime(&that_time);
  (void)snprintf(time_string,BUFFSIZE,"%02d:%02d:%02d %02d/%02d/%4d",
	  time_that->tm_hour,time_that->tm_min,time_that->tm_sec,time_that->tm_mday,(time_that->tm_mon)+1,(time_that->tm_year)+1900);

}
	
/*
  This routine returns the amount of free space on the specified
  path (in KB).
*/
int free_space(char *diskpath,unsigned long long *freespace,unsigned long long *totalspace,char *failure){
  char dfcommand[BUFFSIZE],dfbuf[BUFFSIZE],dfstring[BUFFSIZE],temps[BUFFSIZE],tmp[BUFFSIZE];
  FILE *dfoutput=NULL;
  int tempi,sysretval;
  unsigned long long templ,tempfree,temptotal;

  /* dirty hack alert! */
  (void)snprintf(dfcommand,BUFFSIZE,"df %s > %s/df_result",diskpath,tmp_directory);
  sysretval=system(dfcommand);
  (void)snprintf(tmp,BUFFSIZE,"%s/df_result",tmp_directory);
  if ((dfoutput=fopen(tmp,"r"))==NULL){
    strcpy(failure,FREESPACE_ERR);
    return(FREESPACE);
  }
  while ((fgets(dfbuf,BUFFSIZE,dfoutput))!=NULL)
    strcpy(dfstring,dfbuf);
  fclose(dfoutput);
  remove(tmp);

  /* scan the string read last, it should contain the free space */
  if (sscanf(dfstring,"%s %llu %llu %llu %d%% %s",temps,&temptotal,&templ,&tempfree,&tempi,temps)!=6){
    /* the df string doesn't look like we thought it should */
    strcpy(failure,FREESPACE_ERR);
    return(FREESPACE);
  }

  /* return the free space */
  *freespace=tempfree;
  *totalspace=temptotal;
  return(NO_ERROR);

}

/*
  This routine calculates the current recorder data rate in Mbps.
*/
int recordingdatarate(recorder *settings){
  int nchans,i,temprate,bandwidth;

  nchans=0;
  if (settings->bandwidth<=16)
    bandwidth=settings->bandwidth;
  else
    bandwidth=16; /* Max sampling rate is 32 MHz - this is needed to handle
		     32 and 64 MHz bandwidths */

  for (i=0;i<strlen(settings->compression);i++){ /* figure out the number of channels we're recording */
    if (settings->compression[i]=='x')
      nchans++;
  }
  if ((strcmp(settings->compression,"xo")==0)||
      (strcmp(settings->compression,"ox")==0)){
    /* special mode */
    if (bandwidth<=16)
      nchans*=4; /* actually records 4 channels */
    else
      nchans*=2; /* actually records 2 channels */
  }
  temprate= nchans            /* number of recorded channels */
    * bandwidth*2             /* Nyquist sampling frequency */
    * 2;                      /* 2 bit digitisation */
  if (settings->vsib_mode==2)
    temprate*=2;              /* each sample is 2 bytes */
  return(temprate);
}

/*
  This routine calculates how long (in seconds) it would take to fill the
  amount of free space given the recording data rate.
*/
int recordingfreetime(unsigned long long freespace,int datarate){
  int freetime;
  double bytesfree,bitsfree,timeleft;
  /* the datarate is in Mbps, and the freespace is in KB */
  bytesfree=(double)freespace * 1024.0;
  bitsfree=bytesfree * 8.0;
  timeleft=bitsfree / ((double)datarate * 1e6);
  freetime=(int)timeleft;
  return (freetime);
}

/*
  This routine takes two times, and returns the string representation
  of the time difference.
*/
void time_difference(time_t first_time,time_t second_time,char *string_representation){
  int timediff,daydiff,hourdiff,minutediff,seconddiff;

  timediff=second_time-first_time; /* in seconds */
  daydiff=(int)((float)timediff/86400.0); /* number of days */
  timediff-=daydiff*86400;
  hourdiff=(int)((float)timediff/3600.0); /* hours */
  timediff-=hourdiff*3600;
  minutediff=(int)((float)timediff/60.0); /* minutes */
  timediff-=minutediff*60;
  seconddiff=timediff; /* should be less than 60! */
  if (daydiff>0){
    (void)snprintf(string_representation,BUFFSIZE,"%dd %02d:%02d:%02d",daydiff,hourdiff,minutediff,seconddiff);
  } else {
    (void)snprintf(string_representation,BUFFSIZE,"%02d:%02d:%02d",hourdiff,minutediff,seconddiff);
  }

}

int experiment_add(experiment *newExperiment,char *failure){
  experiment *cycleExperiment=NULL,*oldExperiment=NULL,dummyexperiment;
  int added=0;
/*   struct tm *starttime,*endtime; */
  char tmp[BUFFSIZE],starttime[BUFFSIZE],endtime[BUFFSIZE];

  /* check for an existing experiment with the same name */
  /* this will usually occur if someone tries to load all the profiles */
  /* in a directory more than once - we only want to add new profiles */
  for (cycleExperiment=next_experiment;cycleExperiment!=NULL;cycleExperiment=cycleExperiment->next){
    if (strcmp(newExperiment->experiment_id,cycleExperiment->experiment_id)==0){
      strcpy(failure,PROFILE_SAMEID_ERR);
      return(PROFILE_SAMEID);
    }
  }

  dummyexperiment.start_time=0;
  dummyexperiment.end_time=1;
  oldExperiment=&dummyexperiment;
  for (cycleExperiment=next_experiment;cycleExperiment!=NULL;cycleExperiment=cycleExperiment->next){
    if (newExperiment->start_time<cycleExperiment->start_time){
      /* check that there is no conflict */
      if ((newExperiment->end_time<cycleExperiment->start_time)&&
	  (newExperiment->start_time>oldExperiment->end_time)){
	newExperiment->next=cycleExperiment;
	if (cycleExperiment==next_experiment)
	  /* add at the head of the list */
	  next_experiment=newExperiment;
	added=1;
	if (oldExperiment!=NULL){
	  oldExperiment->next=newExperiment;
	}
	break;
      } else {
	(void)snprintf(tmp,BUFFSIZE,"Conflict in experiment scheduling!");
	PrintStatus(tmp);
	timeasstring(newExperiment->start_time,starttime);
	timeasstring(newExperiment->end_time,endtime);
	(void)snprintf(tmp,BUFFSIZE,"New experiment %s runs from %s to %s",newExperiment->experiment_id,starttime,endtime);
	PrintStatus(tmp);
	if (newExperiment->end_time>cycleExperiment->start_time){
	  timeasstring(cycleExperiment->start_time,starttime);
	  timeasstring(cycleExperiment->end_time,endtime);
	  (void)snprintf(tmp,BUFFSIZE,"Existing experiment %s runs from %s to %s",cycleExperiment->experiment_id,starttime,endtime);
	} else if (newExperiment->start_time<oldExperiment->end_time){
	  timeasstring(oldExperiment->start_time,starttime);
	  timeasstring(oldExperiment->end_time,endtime);
	  (void)snprintf(tmp,BUFFSIZE,"Existing experiment %s runs from %s to %s",oldExperiment->experiment_id,starttime,endtime);
	}
	PrintStatus(tmp);
	strcpy(failure,PROFILE_CONFLICT_ERR);
	return(PROFILE_CONFLICT);
      }
    }
    oldExperiment=cycleExperiment;
  }
  if (added==0){ /* needs to be added on the end of the list */
    /* check that there is no conflict */
    if (next_experiment==NULL){ /* there is no list, this is the first entry */
      newExperiment->next=NULL;
      next_experiment=newExperiment;
    } else if (newExperiment->start_time>oldExperiment->end_time){
      newExperiment->next=NULL;
      oldExperiment->next=newExperiment;
    } else {
      (void)snprintf(tmp,BUFFSIZE,"Conflict in experiment scheduling!");
      PrintStatus(tmp);
      timeasstring(oldExperiment->start_time,starttime);
      timeasstring(oldExperiment->end_time,endtime);
      (void)snprintf(tmp,BUFFSIZE,"Existing experiment %s runs from %s to %s",oldExperiment->experiment_id,starttime,endtime);
      PrintStatus(tmp);
      timeasstring(newExperiment->start_time,starttime);
      timeasstring(newExperiment->end_time,endtime);
      (void)snprintf(tmp,BUFFSIZE,"New experiment %s runs from %s to %s",newExperiment->experiment_id,starttime,endtime);
      PrintStatus(tmp);
      strcpy(failure,PROFILE_CONFLICT_ERR);
      return(PROFILE_CONFLICT);
    }
  }
  return(NO_ERROR);
}

/*
  This routine figures out how to swap disks. It can swap disks
  while recording, or not.
*/
int swapdisk(int flag,recorder *settings,char *failure){
  outputdisk *best_disk=NULL;
  remoterecorder *best_recorder=NULL;
  int old_auto_disk_select,old_execute_experiment,retval;
  char tmpret[BUFFSIZE],remaining_time[BUFFSIZE],pass[BUFFSIZE];
  time_t time_now,time_remaining;

  if (!(flag & DISKSWAP_USESELECTION)){
    /* we're responsible for changing the disk */
    if ((status_server.is_recording==YES)&&
	(status_server.recording_settings->recordingdisk!=NULL)){
      /* we have to exclude the disk that is currently being recorded
         to */
      bestdisk(settings,&best_disk,&best_recorder,
	       status_server.recording_settings->recordingdisk);
    } else {
      /* we just swap to a disk that isn't the one in the passed
         settings */
      bestdisk(settings,&best_disk,&best_recorder,settings->recordingdisk);
    }
    if (best_disk!=NULL){
      settings->recordingdisk=best_disk;
      settings->targetrecorder=best_recorder;
    }
  }

  /* settings now has the disk that we wish to swap to */
  /* change the auto disk selection setting so the recorder won't
     swap it on us again */
  old_auto_disk_select=settings->auto_disk_select;
  settings->auto_disk_select=AUTODISK_DISABLED;

  /* do we have to stop the recording? */
  if ((flag & DISKSWAP_RECORDERCONTROL)&&
      (status_server.is_recording==YES)){
    /* are we running an experiment or manually? */
    if (status_server.experiment_mode==EXPERIMENT_QUEUE){
      /* we're running an experiment */
      old_execute_experiment=status_server.execute_experiment;
      /* stop the experiment */
      status_server.execute_experiment=AUTO_EXPERIMENT_STOP;
      if ((retval=experiment_control(EXPERIMENT_STOP,"",tmpret))!=NO_ERROR){
	(void)snprintf(failure,BUFFSIZE,"could not swap disks: %s",tmpret);
	return(retval);
      }
      /* restart the experiment */
      if ((retval=experiment_control(EXPERIMENT_START,"",tmpret))!=NO_ERROR){
	(void)snprintf(failure,BUFFSIZE,"could not swap disks: %s",tmpret);
	return(retval);
      }
      status_server.execute_experiment=old_execute_experiment;
    } else if (status_server.experiment_mode==EXPERIMENT_MANUAL){
      /* we're running manually, so calculate how much longer we should
         be recording for if we stopped now */
      time(&time_now);
      time_remaining=status_server.recorder_end-time_now;
      /* stop the recording */
      if ((retval=recorder_control(RECSTOP,settings,tmpret))!=NO_ERROR){
	(void)snprintf(failure,BUFFSIZE,"could not swap disks: %s",tmpret);
	return(retval);
      }
      /* set the remaining recording time */
      fuzzy_time(time_remaining,remaining_time);
      (void)snprintf(pass,BUFFSIZE,"record_time=%s",remaining_time);
      /* we don't actually care if this next thing works! */
      data_handler(pass,tmpret,settings);
      /* restart the recorder */
      if ((retval=recorder_control(RECSTART,settings,tmpret))!=NO_ERROR){
	(void)snprintf(failure,BUFFSIZE,"could not swap disks: %s",tmpret);
	return(retval);
      }
    }
  }

  /* set the disk selection back to how it was */
  settings->auto_disk_select=old_auto_disk_select;

  /* we're done */
  return(NO_ERROR);
}

/*
  This routine determines which disk is the best for the current
  recording status.
  It uses the following determination:
  - find the disks that support the current data rate, and satisfies
    the user restrictions on which disks we can choose
  - find the disks that have enough space to contain this entire
    experiment
    + if this can be achieved, select the disk with the smallest
      amount of free space
    + if this cannot be achieved, select the disk with the largest
      amount of free space
*/
void bestdisk(recorder *settings,outputdisk **best_disk,
	      remoterecorder **best_recorder,outputdisk *excludedisk){
  remoterecorder *cyclerecorders=NULL,**potential_recorders=NULL;
  outputdisk *cycledisks=NULL,**potential_disks=NULL;
  int n_potential=0,i,datarate,freetime,requiredtime,singledisk=NO;
  unsigned long long max_freespace=0;

  /* if we're not allowed to make any selection, then we exit
     right now */
  if (settings->auto_disk_select==AUTODISK_DISABLED){
    *best_disk=NULL;
    *best_recorder=NULL;
    return;
  }

  /* assemble the list of disks that can be considered */
  datarate=recordingdatarate(settings);

  /* local disks */
  if ((settings->auto_disk_select==AUTODISK_LOCAL)||
      (settings->auto_disk_select==AUTODISK_ANY)||
      (settings->auto_disk_select==AUTODISK_LIST)){
    for (cycledisks=disk_one;cycledisks!=NULL;
	 cycledisks=cycledisks->next){
      if (cycledisks->is_mounted==NO){
	continue;
      }
      if (cycledisks->max_rate<datarate){
	continue;
      }
      if (cycledisks==excludedisk){
	continue;
      }
      if ((settings->auto_disk_select==AUTODISK_LIST)&&
	  (cycledisks->is_acceptable==NO)){
	continue;
      }
      if (cycledisks==status_server.recording_to_disk){
	continue;
      }
      n_potential++;
      REALLOCV(potential_disks,n_potential*sizeof(outputdisk*));
      potential_disks[n_potential-1]=cycledisks;
      REALLOCV(potential_recorders,n_potential*sizeof(remoterecorder*));
      potential_recorders[n_potential-1]=NULL;
    }
  }
  /* remote disks */
  if ((settings->auto_disk_select==AUTODISK_REMOTE)||
      (settings->auto_disk_select==AUTODISK_ANY)||
      (settings->auto_disk_select==AUTODISK_LIST)){
    for (cyclerecorders=status_server.remote_recorders;
	 cyclerecorders!=NULL;cyclerecorders=cyclerecorders->next){
      for (cycledisks=cyclerecorders->available_disks;
	   cycledisks!=NULL;cycledisks=cycledisks->next){
	if (cycledisks->is_mounted==NO){
	  continue;
	}
	if (cycledisks->max_rate<datarate){
	  continue;
	}
	if (cycledisks==excludedisk){
	  continue;
	}
	if ((settings->auto_disk_select==AUTODISK_LIST)&&
	    (cycledisks->is_acceptable==NO)){
	  continue;
	}
	if (cycledisks==status_server.recording_to_disk){
	  continue;
	}
	n_potential++;
	REALLOCV(potential_disks,n_potential*sizeof(outputdisk*));
	potential_disks[n_potential-1]=cycledisks;
	REALLOCV(potential_recorders,n_potential*sizeof(remoterecorder*));
	potential_recorders[n_potential-1]=cyclerecorders;
      }
    }
  }

  /* if we don't have any disks here we're in trouble! */
  if (n_potential==0){
    /* we can't recommend a disk, so we pass back nothing */
    *best_disk=NULL;
    *best_recorder=NULL;
    return;
  }

  /* go through our list and find the maximum amount of free
     space */
  for (i=0;i<n_potential;i++){
    if ((potential_disks[i])->freespace>max_freespace){
      max_freespace=(potential_disks[i])->freespace;
    }
  }
  /* is this enough free space? */
  freetime=recordingfreetime(max_freespace,datarate);
  requiredtime=timeinseconds(settings->record_time);
  if (freetime<requiredtime){
    singledisk=NO;
  } else {
    singledisk=YES;
  }

  if (singledisk==YES){
    /* find the disk with the smallest amount of space over
       the required time */
    for (i=0;i<n_potential;i++){
      freetime=recordingfreetime((potential_disks[i])->freespace,
				 datarate);
      if (freetime>requiredtime){
	/* we could use this disk */
	if (*best_disk==NULL){
	  *best_disk=potential_disks[i];
	  *best_recorder=potential_recorders[i];
	} else {
	  if ((potential_disks[i])->freespace<(*best_disk)->freespace){
	    *best_disk=potential_disks[i];
	    *best_recorder=potential_recorders[i];
	  }
	}
      }
    }
  } else {
    /* find the disk with the largest amount of space */
    *best_disk=potential_disks[0];
    *best_recorder=potential_recorders[0];
    for (i=1;i<n_potential;i++){
      if ((potential_disks[i])->freespace>(*best_disk)->freespace){
	*best_disk=potential_disks[i];
	*best_recorder=potential_recorders[i];
      }
    }
  }

  /* free our memory */
  FREE(potential_disks);
  FREE(potential_recorders);

}

/*
  This routine is called when the disk space on the recording disk
  falls below the critical level. It can do one of three things:
  1 - do nothing - the recording continues, but will only last for
      another DISK_CRITICAL seconds at maximum
  2 - stop recording - stops the recorder immediately, and will need
      to be restarted manually
  3 - switch disks - stops the recorder, switches the recording disk
      to the one with the most free space, adjusts the recording time
      to stop at the correct time, and restarts the recorder
*/
int disk_critical_action(char *failure){
  int retval;
  char tmpret[BUFFSIZE];

  switch (status_server.low_disk_action){
  case CRITICAL_DONOTHING:
    strcpy(failure,"No action taken.");
    return(NO_ERROR);
    break;
  case CRITICAL_STOP:
    if (status_server.experiment_mode==EXPERIMENT_MANUAL){
      /* just stop the recorder */
      if ((retval=recorder_control(RECSTOP,status_server.current_settings,tmpret))!=NO_ERROR){
	(void)snprintf(failure,BUFFSIZE,"[CRITICAL ACTION] %s",tmpret);
	return(CRITICAL_FAIL);
      }
      IssueWarning("Recorder stopped because disk space low");
      strcpy(failure,"Recording stopped.");
      return(NO_ERROR);
    } else if (status_server.experiment_mode==EXPERIMENT_QUEUE){
      /* we want to allow the user to restart if they want, so
	 switch into manual start for experiments, but allow auto 
         start of subsequent experiments if the user has specified 
         automatic execution and fails to restart this experiment */
      if (status_server.execute_experiment==AUTO_EXPERIMENT_YES)
	status_server.execute_experiment=AUTO_EXPERIMENT_STOP;
      /* stop the experiment */
      if ((retval=experiment_control(EXPERIMENT_STOP,"",tmpret))!=NO_ERROR){
	(void)snprintf(failure,BUFFSIZE,"[CRITICAL ACTION] %s",tmpret);
	return(CRITICAL_FAIL);
      }
      IssueWarning("Recorder stopped because disk space low");      
      strcpy(failure,"Recording stopped.");
      return(NO_ERROR);
    }
    break;
  case CRITICAL_SWITCH:
    /* we call the disk swap routine */
    /* make sure that it can auto-swap the disks */
    if (status_server.recording_settings->auto_disk_select==AUTODISK_DISABLED){
      status_server.recording_settings->auto_disk_select=AUTODISK_ANY;
    }
    /* switch the critical action to stop */
    status_server.low_disk_action=CRITICAL_STOP;

    retval=command_handler("disk-autoswap",tmpret,status_server.recording_settings);
    if (retval!=NO_ERROR){
      (void)snprintf(failure,BUFFSIZE,"[CRITICAL ACTION] %s",tmpret);
      return(CRITICAL_FAIL);
    }

    /* if we get here, we've been successful, so we switch back to switching */
    status_server.low_disk_action=CRITICAL_SWITCH;

    IssueWarning("Recording disk has been switched because disk space low");
    strcpy(failure,"Recording disk switched.");
    return(NO_ERROR);

    break;
  }
  return(NO_ERROR);
}

void fuzzy_time(time_t time_in_seconds,char *representation){
  int nmin,nhour;

  if (time_in_seconds<(time_t)300){ /* five minutes */
    /* output in seconds */
    (void)snprintf(representation,BUFFSIZE,"%ds",(int)time_in_seconds);
  } else if (time_in_seconds<(time_t)3600){ /* one hour */
    /* output in minutes, always go for one extra minute! */
    /* better to run long than pull up short */
    nmin=0;
    while(time_in_seconds>=(time_t)60){
      time_in_seconds-=(time_t)60;
      nmin++;
    }
    nmin++;
    (void)snprintf(representation,BUFFSIZE,"%dm",nmin);
  } else {
    /* if <= 2 minutes away from an integer number of hours, */
    /* then output in hours, else output in minutes */
    nhour=0;
    while(time_in_seconds>=(time_t)3600){
      time_in_seconds-=(time_t)3600;
      nhour++;
    }
    if (time_in_seconds>=(time_t)(3600-120)){
      nhour++;
      time_in_seconds=(time_t)0;
    }
    if (time_in_seconds<=(time_t)120){
      (void)snprintf(representation,BUFFSIZE,"%dh",nhour);
    } else {
      nmin=nhour*60;
      while(time_in_seconds>=(time_t)60){
	time_in_seconds-=(time_t)60;
	nmin++;
      }
      if (time_in_seconds>=(time_t)30)
	nmin++;
      (void)snprintf(representation,BUFFSIZE,"%dm",nmin);
    }
  }      
}

int get_drive_serial(char *serial_numbers,char *filesystem,char *failure,int flag){
  char tmp[BUFFSIZE];
  FILE *serialfile=NULL;
  int sysretval;

  if (flag==DRIVES_SERIAL)
    (void)snprintf(tmp,BUFFSIZE,"%s %s",serial_command,
		   status_server.current_settings->recordingdisk->diskpath);
  else if (flag==DRIVES_LABEL)
    (void)snprintf(tmp,BUFFSIZE,"%s %s --labels --filesystem %s",serial_command,
		   serial_numbers,filesystem);
  //fprintf(stderr,"executing command [%s]\n",tmp);
  sysretval=system(tmp);
  serial_numbers[0]='\0';

  (void)snprintf(tmp,BUFFSIZE,"%s/current_serials",tmp_directory);
  if ((serialfile=fopen(tmp,"r"))==NULL){
    strcpy(failure,SERIAL_FAIL_ERR);
    return(SERIAL_FAIL);
  }
  if ((fgets(tmp,BUFFSIZE,serialfile))==NULL){
    fclose(serialfile);
    strcpy(failure,SERIAL_FAIL_ERR);
    return(SERIAL_FAIL);
  }
  (strstr(tmp,"\n"))[0]='\0'; /* remove newline character */
  strcpy(serial_numbers,tmp);
  fclose(serialfile);
  
  return(NO_ERROR);
}

/*
  This routine conjugates the common name of a host. That is, if
  it doesn't end in "_c" it adds this, and if it does, it removes
  it.
*/
void conjugate_host(char *normal,char *conjugate){
  if (strcmp(normal+strlen(normal)-2,"_c")==0){
    strncpy(conjugate,normal,strlen(normal)-2);
    conjugate[strlen(normal)-2]='\0';
  } else {
    (void)snprintf(conjugate,BUFFSIZE,"%s_c",normal);
  }

}

void copy_recorder(recorder *in,recorder *out){

  /* check that we don't have NULL pointers */
  if ((in==NULL)||(out==NULL)){
    return;
  }

  /* now copy everything over */
  strcpy(out->record_time,in->record_time);
  strcpy(out->record_start_date,in->record_start_date);
  strcpy(out->record_start_time,in->record_start_time);
  strcpy(out->directory_name,in->directory_name);
  strcpy(out->filename_prefix,in->filename_prefix);
  strcpy(out->compression,in->compression);
  out->vsib_mode=in->vsib_mode;
  out->bandwidth=in->bandwidth;
  out->recordingdisk=in->recordingdisk;
  out->targetrecorder=in->targetrecorder;
  out->n_recorders=in->n_recorders;
  out->fringe_test=in->fringe_test;
  strcpy(out->clean_time,in->clean_time);
  strcpy(out->vsib_device,in->vsib_device);
  out->filesize_or_time=in->filesize_or_time;
  out->filesize_is_time=in->filesize_is_time;
  out->blocksize=in->blocksize;
  out->clockrate=in->clockrate;
  out->auto_disk_select=in->auto_disk_select;
  out->mark5b_enabled=in->mark5b_enabled;
  out->udp_enabled=in->udp_enabled;
  out->onebit_enabled=in->onebit_enabled;
  out->numbits=in->numbits;
  out->ipd=in->ipd;

}

void copy_hostent(struct hostent *dest,struct hostent *src){
  /* copies a hostent structure */
  MALLOCV(dest->h_name,BUFFSIZE*sizeof(char));
  strcpy(dest->h_name,src->h_name);
  MALLOCV(dest->h_aliases,sizeof(char *));
  dest->h_aliases[0]=dest->h_name;
  dest->h_addrtype=src->h_addrtype;
  dest->h_length=src->h_length;
  MALLOCV(dest->h_addr_list,sizeof(char *));
  MALLOCV(dest->h_addr_list[0],dest->h_length*sizeof(char));
  strcpy(dest->h_addr_list[0],src->h_addr_list[0]);
  dest->h_addr=dest->h_addr_list[0];
}

void free_hostent(struct hostent *dest){
  FREE(dest->h_name);
  FREE(dest->h_aliases);
  FREE(dest->h_addr_list[0]);
  FREE(dest->h_addr_list);
}

/*
  This routine takes the current settings and copies them into a
  target recorder structure, assigns the id (overwriting an entry with
  the same id if necessary) and then adds it to the target list.
*/
int copy_recordertarget(int flag,char *target_id,recorder *settings,char *failure){
  rectarget *cycle_targets=NULL,*new_target=NULL,*free_target=NULL,*old_target=NULL;

  if (flag==RECORDERTARGET_MAKE){

    /* determine whether there is a recording target with the same
       identifier already */
    for (cycle_targets=status_server.target_list;cycle_targets!=NULL;
	 cycle_targets=cycle_targets->next){
      if (strcmp(cycle_targets->target_identifier,target_id)==0){
	/* we might need to overwrite this target, but first we check
	   if it is currently involved in a recording */
	if (cycle_targets->is_recording==YES){
	  /* can't overwrite settings of a currently recording target */
	  strcpy(failure,RECTARG_USEDCOPY_ERR);
	  return(RECTARG_USEDCOPY);
	}
	new_target=cycle_targets;
	break;
      }
    }
    
    /* do we need to create a new target? */
    if (new_target==NULL){
      MALLOC(new_target,sizeof(rectarget));
      new_target->recorder_settings=NULL;
      new_target->is_recording=NO;
      strcpy(new_target->target_identifier,target_id);
      /* add it to the list */
      new_target->next=status_server.target_list;
      status_server.target_list=new_target;
    }
    
    /* copy the recording settings */
    /* do we need to allocate some space for the settings? */
    if (new_target->recorder_settings==NULL){
      MALLOC(new_target->recorder_settings,sizeof(recorder));
    }
    copy_recorder(settings,new_target->recorder_settings);

  } else if (flag==RECORDERTARGET_RECALL){
    /* copy the target settings back to the current settings */
    for (cycle_targets=status_server.target_list;cycle_targets!=NULL;
	 cycle_targets=cycle_targets->next){
      if (strcmp(cycle_targets->target_identifier,target_id)==0){
	/* we've found the right target */
	new_target=cycle_targets;
	break;
      }
    }

    if (new_target==NULL){
      /* couldn't find the specified target! */
      strcpy(failure,RECTARG_NOTFOUND_ERR);
      return(RECTARG_NOTFOUND);
    }
    /* copy the recording settings */
    copy_recorder(new_target->recorder_settings,settings);

  } else if (flag==RECORDERTARGET_REMOVE){
    /* look for the named target */
    for (cycle_targets=status_server.target_list;
	 cycle_targets!=NULL;cycle_targets=cycle_targets->next){
      if (strcmp(cycle_targets->target_identifier,target_id)==0){
	/* this is the target to remove */
	if (cycle_targets->is_recording==YES){
	  /* can't remove it, it's currently involved in a 
	     recording */
	  strcpy(failure,RECTARG_REMUSED_ERR);
	  return(RECTARG_REMUSED);
	}
	free_target=cycle_targets;
	/* remove it from the linked list */
	if (old_target==NULL){
	  /* it was the first target */
	  status_server.target_list=cycle_targets->next;
	} else {
	  old_target->next=cycle_targets->next;
	}
	break;
      }
      old_target=cycle_targets;
    }

    /* free the memory */
    FREE(free_target->recorder_settings);
    FREE(free_target);

  }

  /* all is well */
  return(NO_ERROR);
  
}
