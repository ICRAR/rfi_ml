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


/*
RCS info
                                                                                
$Id: recorder_server.h,v 1.18 2010-10-24 22:47:28 ste616 Exp $
$Source: /epp/atapplic/cvsroot/evlbi/recorder_server/recorder_server.h,v $
*/

#define header_version "$Id: recorder_server.h,v 1.18 2010-10-24 22:47:28 ste616 Exp $"

/* some magic numbers */
#define LISTENPORT 50080  /* the port to listen on for commands */
#define MAXPENDING 5      /* Max connection requests */
#define BUFFSIZE 10000    /* The largest message that can be sent to this server */
#define MAXCHECKARGS 3    /* the maximum number of arguments that can be sent from command_handler 
			     to check_control */
#define MAXCHECKALLOW 5   /* the number of arguments command_handler recognises for check-* */
#define STATUS_UPDATE 20  /* the number of seconds between automatic status updates and warning checks */
#define EXPSWITCHTIME 5   /* the minimum time (in seconds) needed before the experiment can be started */
#define EXPUPDATEBLK  10  /* add this number to STATUS_UPDATE to get the number of seconds before an 
			     experiment starts that the server will set SIGALRM for experiment start */
#define EXPSTARTBUF   5   /* number of seconds between vsib_record start and experiment start */
#define DISK_WARNING1 180 /* first warning when number of seconds remaining on disk is less than this */
#define DISK_WARNING2 60  /* second warning when number of seconds remaining on disk is less than this */
#define DISK_CRITICAL 30  /* critical action taken when number of seconds remaining on disk is less than this */
#define MAX_COMMS_FAIL 10 /* accept this number of communications failures to a remote host before removing it */

			     
#define DEBUG_MESSAGES 1  /* set this to 1 to see debugging messages, otherwise set to 0 */



/* error code definitions */
#define GENERAL_ERROR     -1 /* something went wrong */

#define NO_ERROR           0 /* everything worked */

#define SHORT_MESG         1 /* the message sent to the server is too short */
#define SHORT_MESG_ERR "message too short"

#define MALFORMED_DATA     2 /* the data request does not follow variable=data*/
#define MALFORMED_DATA_ERR "malformed data request"

#define MALFORMED_DATE     3 /* the start date does not follow either yyyy-mm-dd or yyyymmdd */
#define MALFORMED_DATE_ERR "malformed starting date"

#define MONTH_RANGE        4 /* the start month is not valid */
#define MONTH_RANGE_ERR "the month must be a number between 1 and 12"

#define DATE_RANGE         5 /* the start day-of-month is not valid */
#define DATE_RANGE_ERR  "the day-of-month must be a number between 1 and 31, and be valid for the supplied month"

#define IN_THE_PAST        6 /* the starting date is in the past */
#define IN_THE_PAST_ERR "the starting date/time is in the past"

#define MALFORMED_TIME     7 /* the start time does not follow either hh:mm:ss or hhmmss */
#define MALFORMED_TIME_ERR "malformed starting time"

#define DATE_NOT_SET       8 /* the starting date must be set before the starting time */
#define DATE_NOT_SET_ERR "the starting date must be set before setting the starting time"

#define HOUR_RANGE         9 /* the start hour is not valid */
#define HOUR_RANGE_ERR "the hour must be a number between 0 and 23"

#define MINUTE_RANGE      10 /* the start minute is not valid */
#define MINUTE_RANGE_ERR "the minute must be a number between 0 and 59"

#define SECOND_RANGE      11 /* the start second is not valid */
#define SECOND_RANGE_ERR "the second must be a number between 0 and 59"

#define BAD_COMPRESSION   12 /* the compression string is not valid */
#define BAD_COMPRESSION_ERR "the compression settings are not valid"

#define BAD_VSIB_MODE     13 /* the vsib mode is not valid */
#define BAD_VSIB_MODE_ERR "the specified vsib mode is not valid"

#define BAD_BANDWIDTH     14 /* the bandwidth is not valid */
#define BAD_BANDWIDTH_ERR "the specified bandwidth is not valid"

#define BAD_DATA          15 /* the data variable is not valid */
#define BAD_DATA_ERR "the specified data variable is not valid"

#define INVALID_S2_MODE   16 /* the specified S2 mode is invalid */
#define INVALID_S2_MODE_ERR "the specified mode of the S2 recorder is not valid"

#define UNKNOWN_COMMAND   17 /* the command is not valid */
#define UNKNOWN_COMMAND_ERR "the command issued is not valid"

#define FILE_UNREADABLE   18 /* the requested file was not able to be read */
#define FILE_UNREADABLE_ERR "unable to open the recorder settings file for reading"

#define FILE_UNWRITABLE   19 /* the requested file was not able to be written */
#define FILE_UNWRITABLE_ERR "unable to open the recorder settings file for writing"

#define ALREADY_RECORDING 20 /* a request to start recording was made while the recorder was running */
#define ALREADY_RECORDING_ERR "request to start recording denied: recorder is already running"

#define RECORDING_TIME    21 /* the requested recording time is too low */
#define RECORDING_TIME_ERR "the recording time has not been set"

#define DIRECTORY_NOT_SET 22 /* the directory name has not been set */
#define DIRECTORY_NOT_SET_ERR "the output directory has not been specified"

#define FILENAME_NOT_SET  23 /* the filename prefix has not been set */
#define FILENAME_NOT_SET_ERR "the output filename prefix has not been specified"

#define DIRECTORY_INVALID 24 /* the name of the directory cannot be used */
#define DIRECTORY_INVALID_ERR "the name specified for the output directory is being used by a non-directory"

#define MKDIR_FAILED      25 /* the output directory was unable to be created */
#define MKDIR_FAILED_ERR "the output directory was unable to be created"

#define CHDIR_FAILED      26 /* could not chdir into the output directory */
#define CHDIR_FAILED_ERR "could not chdir into the output directory"

#define SETTINGS_GONE     27 /* the recorder settings file was not written, or was deleted */
#define SETTINGS_GONE_ERR "the recorder settings file was not written as expected, or has been deleted"

#define UNEXPECTED_EOF    28 /* there was not enough information in the recorder settings file */
#define UNEXPECTED_EOF_ERR "the recorder settings file is corrupt"

#define CORRUPT_SETTINGS  29 /* the recorder settings file was not accepted by the data handler */
#define CORRUPT_SETTINGS_ERR "the recorder settings file has been corrupted"

#define NOT_RECORDING     30 /* a request was made to stop the recorder, but recording has not started */
#define NOT_RECORDING_ERR "request to stop recording denied: recorder not currently running"

#define STOP_FAILED       31 /* the stop recorder request failed */
#define STOP_FAILED_ERR "unable to stop recorder process"

#define MALFORMED_REQUEST 32 /* the request to the server was not specified correctly */
#define MALFORMED_REQUEST_ERR "malformed request"

#define RESET_FAILED      33 /* the requested data could not be reset to defaults */
#define RESET_FAILED_ERR "the requested data could not be reset to its default value"

#define INVALID_RESET     34 /* cannot reset data variable because it doesn't exist */
#define INVALID_RESET_ERR "cannot reset data variable because it doesn't exist"

#define MALFORMED_RECTIME 37 /* the recording time doesn't look like it should */
#define MALFORMED_RECTIME_ERR "the specified recording time is invalid"

#define FREESPACE         38 /* the free space on the disk cannot be determined */
#define FREESPACE_ERR "the amount of free space on the disk cannot be determined"

#define STATUS_INVALID    39 /* requested an unrecognised status mode */
#define STATUS_INVALID_ERR "the requested status mode is invalid"

#define STATUS_COMPLETE   40 /* the status request was completed successfully - a dummy error code */

#define NO_DISKS          43 /* no recording disks were specified */
#define NO_DISKS_ERR "no recording disks were specified in the configuration file"

#define DISKNOTFOUND      44 /* the requested disk number was not available */
#define DISKNOTFOUND_ERR "the requested recording disk number is not valid"

#define FRINGE_TEST_VAL   46 /* invalid fringe-test option, must be yes or no */
#define FRINGE_TEST_VAL_ERR "invalid specification of fringe-test mode: must be yes or no"

#define FRINGETEST_RESET  47 /* cannot reset fringe-test mode */
#define FRINGETEST_RESET_ERR "cannot reset fringe-test mode: it must be explicitly set to yes or no"

#define STATUS_WRITE_FAIL 48 /* cannot write the current server status to disk */
#define STATUS_WRITE_FAIL_ERR "unable to open file to output current server status"

#define RECORDER_OFF      49 /* asked to write recorder settings, but not recording */
#define RECORDER_OFF_ERR "cannot write recorder settings before recorder started"

#define STATUS_READ_FAIL  50 /* cannot read the server status from disk */
#define STATUS_READ_FAIL_ERR "unable to open file to input stored server status"

#define STATUS_CORRUPT    51 /* the server status file does not look like it should */
#define STATUS_CORRUPT_ERR "unable to properly read the server status file"

#define PROFILE_NOT_FOUND 52 /* the specified experiment profile could not be loaded */
#define PROFILE_NOT_FOUND_ERR "unable to read specified experiment profile - check location and permissions"

#define PROFILE_CORRUPT   53 /* the experiment profile is not formatted correctly */
#define PROFILE_CORRUPT_ERR "the specified experiment profile is not properly formatted"

#define PROF_STTIME_INV   54 /* the start time of the profile is invalid */
#define PROF_STTIME_INV_ERR "the start time of the experiment profile must be formatted as HH:MM:SS dd/mm/yyyy"

#define PROF_STTIME_PAST  55 /* the start time of the profile is in the past */
#define PROF_STTIME_PAST_ERR "the start time of the experiment profile is in the past"

#define PROF_ENTIME_INV   56 /* the end time of the profile is invalid */
#define PROF_ENTIME_INV_ERR "the end time of the experiment profile must be formatted as HH:MM:SS dd/mm/yyyy"

#define PROF_ENTIME_PAST  57 /* the end time of the profile is in the past */
#define PROF_ENTIME_PAST_ERR "the end time of the experiment profile is in the past"

#define PROF_TIME_REVERSE 58 /* the start time is after the end time */
#define PROF_TIME_REVERSE_ERR "the start time must be before the end time in the experiment profile"

#define PROFILE_INVALID   59 /* the experiment profile does not have all the required information */
#define PROFILE_INVALID_ERR "not all the required information was available in the experiment profile"

#define PROFILE_NOSTART   60 /* can't start the experiment profile because the server was started too late */
#define PROFILE_NOSTART_ERR "the experiment cannot be started due to the late start of the server"

#define PROFILE_CONFLICT  61 /* a scheduling conflict exists between two profiles */
#define PROFILE_CONFLICT_ERR "the profile could not be loaded, as it conflicts in scheduling with another experiment"

#define LOAD_NOT_RUNNING  62 /* tried to load the recorder settings while the recorder wasn't running */
#define LOAD_NOT_RUNNING_ERR "should not load the recorder settings when the recorder is not running"

#define RECORD_FORK_FAIL  63 /* couldn't fork to start the recorder */
#define RECORD_FORK_FAIL_ERR "unable to fork process to begin recorder"

#define RECORD_START_FAIL 64 /* the recorder did not start properly */
#define RECORD_START_FAIL_ERR "the recorder did not start correctly"

#define PROFILE_DIR_BAD   65 /* the location of the experiment dir is invalid */
#define PROFILE_DIR_BAD_ERR "the location of the experiment profile directory is not known"

#define PROFILE_DIR_READ  66 /* couldn't see the files in the experiment dir */
#define PROFILE_DIR_READ_ERR "unable to see the files in the experiment profile directory"

#define PROFILE_DIR_EMPTY 67 /* there are no files in the experiment dir */
#define PROFILE_DIR_EMPTY_ERR "there are no files in the experiment profile directory"

#define BAD_PROFILE_CMD   68 /* the requested experiment command is invalid */
#define BAD_PROFILE_CMD_ERR "the requested experiment command is invalid"

#define NO_PROFILE_GIVEN  69 /* a load/unload request was made, but no profile specified */
#define NO_PROFILE_GIVEN_ERR "you must specify an experiment profile to load/unload"

#define EXPERIMENT_OVER   70 /* tried to start experiment after its end time */
#define EXPERIMENT_OVER_ERR "cannot start an experiment after its end time"

#define EXP_QUEUE_DONE    71 /* there are no more experiments in the queue */
#define EXP_QUEUE_DONE_ERR "there are no more experiments in the queue"

#define EXP_START_WRONG   72 /* can't start an experiment because we've been asked wrongly */
#define EXP_START_WRONG_ERR "cannot start experiment - wrong mode"

#define EXP_START_LONG    73 /* the start time of the new experiment is too far away */
#define EXP_START_LONG_ERR "new experiment not started as start time is too far away"

#define PROFILE_SAMEID    74 /* attempt to load two experiments with the same id */
#define PROFILE_SAMEID_ERR "cannot load the experiment as experiment with the same ID name already loaded"

#define RECORDDISK_NONE   75 /* no recording disk was specified when starting the recorder */
#define RECORDDISK_NONE_ERR "no recording disk has been specified"

#define NO_UNLOAD_MATCH   76 /* tried to remove an experiment that wasn't loaded */
#define NO_UNLOAD_MATCH_ERR "that experiment is not in the queue"

#define PROFILE_MISMATCH  77 /* the experiment id is not the same as the filename */
#define PROFILE_MISMATCH_ERR "the experiment ID is not the same as the profile filename"

#define MANUAL_AUTO_ON    78 /* tried to start/stop an experiment while auto control was on */
#define MANUAL_AUTO_ON_ERR "cannot permit manual experiment control while auto experiment control is enabled"

#define EXPSAVE_TOOMANY   79 /* too many arguments were supplied to the experiment-save command */
#define EXPSAVE_TOOMANY_ERR "more than one argument was supplied for experiment-save"

#define EXPSAVE_DIRNAME   80 /* neither a directory name nor a profile name was supplied */
#define EXPSAVE_DIRNAME_ERR "must supply either a directory name or a name to save the experiment profile as"

#define EXPSAVE_EXISTS    81 /* can't write over an existing profile */
#define EXPSAVE_EXISTS_ERR "unable to overwrite existing experiment profile"

#define EXPSAVE_NOWRITE   82 /* can't write to the experiment profile directory */
#define EXPSAVE_NOWRITE_ERR "unable to create experiment profile in the profile directory"

#define PROF_NOTCURRENT   83 /* the loaded experiment doesn't match the current recorder settings */
#define PROF_NOTCURRENT_ERR "the current recorder settings do not match this experiment's profile"

#define AUTOCONTROL_VAL   84 /* the auto experiment control mode is specified incorrectly */
#define AUTOCONTROL_VAL_ERR "auto experiment control must be specified as either on or off"

#define NO_IMMEDIATE      85 /* tried to load an "immediate" profile while an experiment was running */
#define NO_IMMEDIATE_ERR "cannot load an immediate profile while an experiment is running"

#define CRITICAL_FAIL     86 /* the automatic action when disk space is low failed - various messages */

#define INVALIDDISKACTION 87 /* the specified disk action is not recognised */
#define INVALIDDISKACTION_ERR "the critical disk action is one of none, stop or swap"

#define DATARATE_TOOFAST  88 /* the selected disk can't handle the datarate */
#define DATARATE_TOOFAST_ERR "the recording data-rate cannot be handled by the selected disk"

#define NO_STATION_NAME   89 /* the station name wasn't found in the config file */
#define NO_STATION_NAME_ERR "the name of the station was not included in the /etc/recorder_disks.conf file"

#define EXPSAVE_FCHECK    90 /* the experiment ID must be specified for Fringe Check experiments */
#define EXPSAVE_FCHECK_ERR "the experiment ID must be specified when saving fringe-check experiments"

#define HEALTH_BAD_STRING 94 /* the health string passed to the server was malformed */
#define HEALTH_BAD_STRING_ERR "malformed recorder health string received"

#define HEALTH_BAD_CMND   95 /* the command sent to the health control routine is unknown */
#define HEALTH_BAD_CMND_ERR "unknown command sent to health_control"

#define HEALTH_FORK_FAIL  96 /* unable to start forked process to start health checker */
#define HEALTH_FORK_FAIL_ERR "unable to start forked process to start health checker"

#define HEALTH_RUNNING    97 /* there is already a health checker process */
#define HEALTH_RUNNING_ERR "cannot start new health checker as one already exists"

#define HEALTH_NOTRUNNING 98 /* no health checker process running */
#define HEALTH_NOTRUNNING_ERR "unable to stop health checker as no process exists"

#define HEALTH_NORECORDER 99 /* no recorder running when requesting health checker start */
#define HEALTH_NORECORDER_ERR "cannot start new health checker as no recorder is running"

#define ROUNDED_INVALID  104 /* specified an invalid 10s boundary mode */
#define ROUNDED_INVALID_ERR "invalid mode specified for auto-10s rounding option"

#define EXP_ROUNDED_BAD  105 /* unknown value for the TEN_SECOND parameter in the experiment file */
#define EXP_ROUNDED_BAD_ERR "must specify yes or no for TEN_SECOND parameter in experiment profiles"

#define SERIAL_FAIL      106 /* the command to get the drive's serial number failed */
#define SERIAL_FAIL_ERR "could not determine serial number of recording disk"

#define UNLOAD_CURRENT   109 /* can't unload currently executing experiment */
#define UNLOAD_CURRENT_ERR "cannot unload the experiment, as it is currently being executed"

#define MALLOC_FAILED    110
#define REALLOC_FAILED   111

#define COMPRESS_INVALID 112
#define COMPRESS_INVALID_ERR "recorder does not support specified compression scheme"

#define EVLBI_INVALID    113
#define EVLBI_INVALID_ERR "invalid mode for eVLBI operation"

#define REMPORT_INVALID  114
#define REMPORT_INVALID_ERR "remote port must be between 1 and 65535 inclusive"

#define EVLBI_UNKNOWN    115
#define EVLBI_UNKNOWN_ERR "request for eVLBI recording cannot proceed as connection info not given"

#define CHDEFAULT_FAIL   116 /* couldn't change into the default directory */
#define CHDEFAULT_FAIL_ERR "unable to change into default directory"

#define CHDEFAULT_RUN    117 /* something was running when we were asked to change to the default directory*/
#define CHDEFAULT_RUN_ERR "stop all recordings/cleaners before changing to default directory"

#define RECV_RUNNING     118 /* the receiver was asked to start but is already running */
#define RECV_RUNNING_ERR "a receiver process is already running"

#define RECV_STOPPED     119 /* the receiver was asked to stop but was not running */
#define RECV_STOPPED_ERR "no receiver process is running"

#define RECV_EVLBI_OFF   120 /* remote recorder is unknown before starting a receiver */
#define RECV_EVLBI_OFF_ERR "must specify remote recorder before starting receiver"

#define RECV_REMPORT     121 /* no remote port set for receiver start */
#define RECV_REMPORT_ERR "you must specify a remote TCP port before starting receiver"

#define RECV_TCPWINDOW   122 /* no TCP window size set for receiver start */
#define RECV_TCPWINDOW_ERR "you must specify a TCP window size before starting receiver"

#define RECV_FORK_FAIL   123 /* couldn't fork to start the receiver */
#define RECV_FORK_FAIL_ERR "unable to fork process to begin receiver"

#define RECV_START_FAIL  124 /* the receiver did not start properly */
#define RECV_START_FAIL_ERR "the receiver did not start correctly"

#define RECV_STOP_FAILED 125 /* the stop receiver request failed */
#define RECV_STOP_FAILED_ERR "unable to stop receiver process"

#define MARK5B_INVALID   128 /* didn't specify yes or no for Mark5B operation */
#define MARK5B_INVALID_ERR "invalid mode specified for Mark5B operation"

#define UDP_INVALID      129 /* didn't specify UDP MTU size */
#define UDP_INVALID_ERR "invalid MTU specified for UDP operation"

#define IFC_NONAMES      130 /* couldn't get network interface names */
#define IFC_NONAMES_ERR "unable to get network interface names"

#define ONEBIT_INVALID   131 /* didn't recognise 1 bit recording mode */
#define ONEBIT_INVALID_ERR "invalid mode specified for 1 bit recording"

#define HOST_SPECBAD     132 /* couldn't understand the remote host specification */
#define HOST_SPECBAD_ERR "could not understand the remote host string"

#define HOST_LOOKUPFAIL  133 /* couldn't get an IP for the remote host */
#define HOST_LOOKUPFAIL_ERR "unable to resolve specified remote host"

#define DISK_SPECBAD     134 /* couldn't understand the remote disk specification */
#define DISK_SPECBAD_ERR "could not understand the remote disk string - don't manually use this option!"

#define UNKNOWN_REMHOST  135 /* have not been told about the remote host that is trying to contact us */
#define UNKNOWN_REMHOST_ERR "unknown remote host"

#define UNMATCHED_DISK   137 /* the remote disk selected does not belong to the target recorder */
#define UNMATCHED_DISK_ERR "please select the remote disk again - an internal error may have occurred"

#define HOST_MODIFYBAD   138 /* couldn't understand the remote host modification string */
#define HOST_MODIFYBAD_ERR "could not understand the remote host modification string"

#define HOST_REMSPECBAD  139 /* couldn't understand the remove host string */
#define HOST_REMSPECBAD_ERR "could not understand the remove remote host string"

#define REMHOST_UNKNOWN  140 /* do not know about the host we've been asked to remove */
#define REMHOST_UNKNOWN_ERR "could not remove unknown remote host"

#define RECV_NORECORDERS 141 /* no recorders were specified before receive command was given */
#define RECV_NORECORDERS_ERR "must specify at least one recorder before starting the receiver"

#define RECV_RECNOTSET   142 /* no recorder was specified in receive command */
#define RECV_RECNOTSET_ERR "no recorder was specified in the receive command"

#define HOST_REMUSED     143 /* the remote host is being used so can't be removed */
#define HOST_REMUSED_ERR "cannot remove a host that is being used"

#define RECTARG_NONE     144 /* have not specified a recorder target identifier */
#define RECTARG_NONE_ERR "no recorder target identifier was specified"

#define RECTARG_USEDCOPY 145 /* can't overwrite settings of a target that is currently being recorded to */
#define RECTARG_USEDCOPY_ERR "unable to overwrite recorder target - it is currently being recorded to"

#define REMREC_UNKNOWN   146 /* can't find the specified remote recorder */
#define REMREC_UNKNOWN_ERR "unable to find remote recorder with the specified name"

#define RECTARG_NOTFOUND 147 /* couldn't find the specified recorder target */
#define RECTARG_NOTFOUND_ERR "unable to find the target with the specified identifier"

#define DISKSEL_UNKNOWN  148 /* invalid value specified for diskselection */
#define DISKSEL_UNKNOWN_ERR "auto disk selection method not recognised"

#define RECTARG_REMUSED  149 /* can't remove a recorder target that is currently being recorded to */
#define RECTARG_REMUSED_ERR "unable to remove recorder target - it is currently being recorded to"

/* recorder control flags */
#define RECSTART  -10 /* starts the recorder */
#define RECSTOP   -20 /* stops the recorder */

/* status checking flags */
#define STATUS_RECORD      -50  /* check the status of the recorder */
#define STATUS_SERVER      -70  /* check the status of the server */
#define STATUS_SETTINGS    -80  /* list the current recorder settings */
#define STATUS_IFCONFIG    -100 /* list the static interface configuration */
#define STATUS_NETWORK     -110 /* show network activity */
#define STATUS_RECEIVER    -120 /* check the status of the receiver */
#define STATUS_REMOTEHOSTS -121 /* get a list of remote hosts known to the server */
#define STATUS_RECSETTINGS -122 /* list the recorder settings that started the
				   current recording */
#define STATUS_TARGETS     -123 /* list the recording targets and their settings */
#define STATUS_USERDISKS   -124 /* list the user-selected acceptable disks */

/* experiment control flags */
#define EXPERIMENT_LOAD   -130 /* loads an experiment */
#define EXPERIMENT_SAVE   -140 /* saves manual settings as an experiment */
#define EXPERIMENT_UNLOAD -150 /* removes and experiment into the queue */
#define EXPERIMENT_LIST   -160 /* lists the experiments on disk */
#define EXPERIMENT_START  -170 /* start an experiment manually */
#define EXPERIMENT_STOP   -180 /* stop an experiment manually */

/* experiment ready call flags */
#define READY_AUTO   -190 /* the server has called for an experiment to begin */
#define READY_MANUAL -200 /* the user has called for an experiment to begin */

/* recorder error flags */
#define RECORDER_OK    -210 /* the recorder is fine */
#define RECORDER_ERROR -220 /* the recorder has encountered an error */

/* server status switches */
#define CRITICAL_DONOTHING   -230
#define CRITICAL_STOP        -240
#define CRITICAL_SWITCH      -250
#define EXPERIMENT_MANUAL    -260
#define EXPERIMENT_QUEUE     -270
#define AUTO_EXPERIMENT_YES  -280
#define AUTO_EXPERIMENT_NO   -290
#define AUTO_EXPERIMENT_STOP -300
#define WARNING_NONE         -310
#define WARNING_ONE          -320
#define WARNING_TWO          -330
#define WARNING_CRITICAL     -340
#define YES                   1
#define NO                    0
#define BAD                  -1

/* health checker flags */
#define HEALTH_START  -350
#define HEALTH_STOP   -360

/* rounded boundary conditions */
#define ROUNDSTART_YES  -390
#define ROUNDSTART_NO   -400

/* socket_options calling flags */
#define SOCKETOPTIONS_MAIN  -410
#define SOCKETOPTIONS_OTHER -420

/* health handler flags */
#define HEALTH_INFO  -430
#define HEALTH_ERROR -440
#define HEALTH_TIME  -450

/* drive label flags */
#define DRIVES_SERIAL -460
#define DRIVES_LABEL  -470

/* remote disks flags */
#define REMOTEDISK_LIST   -480
#define REMOTEDISK_STATUS -490

/* eVLBI receiver flags */
#define RECEIVER_START -500
#define RECEIVER_STOP  -510

/* remote host removal flags */
#define REMOTEHOST_REMOVE_IMMEDIATE -520
#define REMOTEHOST_REMOVE_COMMSFAIL -530

/* automatic disk selection flags */
#define AUTODISK_DISABLED -540
#define AUTODISK_ANY      -550
#define AUTODISK_LOCAL    -560
#define AUTODISK_REMOTE   -570
#define AUTODISK_LIST     -580

/* recorder target flags */
#define RECORDERTARGET_MAKE   -590
#define RECORDERTARGET_RECALL -600
#define RECORDERTARGET_REMOVE -610

/* disk swapping flags */
#define DISKSWAP_RECORDERCONTROL  0x1
#define DISKSWAP_THREADCONTROL    0x2
#define DISKSWAP_USESELECTION     0x4

/* data reset flags */
#define RESET_RECORD_TIME         1
#define RESET_RECORD_START_DATE   2
#define RESET_RECORD_START_TIME   4
#define RESET_DIRECTORY_NAME      8
#define RESET_FILENAME_PREFIX    16
#define RESET_COMPRESSION        32
#define RESET_VSIB_MODE          64
#define RESET_BANDWIDTH         128
#define RESET_DISKS             256
#define RESET_DEVICE           1024
#define RESET_FILESIZE         2048
#define RESET_BLOCKSIZE        4096
#define RESET_CLOCKRATE        8192
#define RESET_MARK5B         262144
#define RESET_NUMBITS        524288
#define RESET_ENCODING      1048576
#define RESET_FREQUENCY     2097152
#define RESET_POLARISATION  4194304
#define RESET_SIDEBAND      8388608
#define RESET_REFERENCEANT 0x1000000
#define RESET_UDP          0x2000000
#define RESET_IPD          0x4000000
#define RESET_AUTODISK     0x8000000
#define RESET_ALL          0xFFFFFFF
#define RESET_INITIAL      0x1FFFFFFF

/* allowable compression modes */
#define COMPRESSION_CHAN1   0x1
#define COMPRESSION_CHAN2   0x2
#define COMPRESSION_CHAN3   0x4
#define COMPRESSION_CHAN4   0x8
#define COMPRESSION_XO      0x10
#define COMPRESSION_OX      0x20
#define COMPRESSION_BW16    0x40
#define COMPRESSION_BW32    0x80
#define COMPRESSION_BW64    0x100
#define COMPRESSION_MODE3   0x200
#define COMPRESSION_MODE2   0x400
#define COMPRESSION_XXOOOOXX 0x800
#define COMPRESSION_OOXXXXOO 0x1000

static int allowed_compression_modes[] =
  {
    COMPRESSION_CHAN1 | COMPRESSION_MODE3 | COMPRESSION_BW16,
    COMPRESSION_CHAN2 | COMPRESSION_MODE3 | COMPRESSION_BW16,
    COMPRESSION_CHAN3 | COMPRESSION_MODE3 | COMPRESSION_BW16,
    COMPRESSION_CHAN4 | COMPRESSION_MODE3 | COMPRESSION_BW16,
    COMPRESSION_CHAN1 | COMPRESSION_CHAN2 | COMPRESSION_MODE3 | COMPRESSION_BW16,
    COMPRESSION_CHAN1 | COMPRESSION_CHAN3 | COMPRESSION_MODE3 | COMPRESSION_BW16,
    COMPRESSION_CHAN3 | COMPRESSION_CHAN4 | COMPRESSION_MODE3 | COMPRESSION_BW16,
    COMPRESSION_CHAN2 | COMPRESSION_CHAN4 | COMPRESSION_MODE3 | COMPRESSION_BW16,
    COMPRESSION_CHAN1 | COMPRESSION_CHAN2 | COMPRESSION_CHAN3 | COMPRESSION_CHAN4 | COMPRESSION_MODE3 | COMPRESSION_BW16,
    COMPRESSION_XO | COMPRESSION_MODE2 | COMPRESSION_BW16,
    COMPRESSION_XXOOOOXX | COMPRESSION_MODE2 | COMPRESSION_BW16,
    COMPRESSION_OOXXXXOO | COMPRESSION_MODE2 | COMPRESSION_BW16,
    COMPRESSION_CHAN2 | COMPRESSION_MODE2 | COMPRESSION_BW16,
    COMPRESSION_CHAN4 | COMPRESSION_MODE2 | COMPRESSION_BW16,
    COMPRESSION_CHAN1 | COMPRESSION_CHAN3 | COMPRESSION_MODE2 | COMPRESSION_BW16,
    COMPRESSION_CHAN2 | COMPRESSION_CHAN4 | COMPRESSION_MODE2 | COMPRESSION_BW16,
    COMPRESSION_CHAN1 | COMPRESSION_CHAN2 | COMPRESSION_CHAN3 | COMPRESSION_CHAN4 | COMPRESSION_MODE2 | COMPRESSION_BW16,
    COMPRESSION_CHAN1 | COMPRESSION_CHAN2 | COMPRESSION_MODE2 | COMPRESSION_BW16,
    COMPRESSION_CHAN3 | COMPRESSION_CHAN4 | COMPRESSION_MODE2 | COMPRESSION_BW16,
    COMPRESSION_CHAN1 | COMPRESSION_CHAN2 | COMPRESSION_MODE3 | COMPRESSION_BW32,
    COMPRESSION_CHAN3 | COMPRESSION_CHAN4 | COMPRESSION_MODE3 | COMPRESSION_BW32,
    COMPRESSION_CHAN1 | COMPRESSION_CHAN2 | COMPRESSION_CHAN3 | COMPRESSION_CHAN4 | COMPRESSION_MODE3 | COMPRESSION_BW32,
    COMPRESSION_CHAN1 | COMPRESSION_CHAN2 | COMPRESSION_MODE2 | COMPRESSION_BW32,
    COMPRESSION_CHAN3 | COMPRESSION_CHAN4 | COMPRESSION_MODE2 | COMPRESSION_BW32,
    COMPRESSION_CHAN1 | COMPRESSION_CHAN2 | COMPRESSION_CHAN3 | COMPRESSION_CHAN4 | COMPRESSION_MODE2 | COMPRESSION_BW32,
    COMPRESSION_XO | COMPRESSION_MODE2 | COMPRESSION_BW32,
    COMPRESSION_OX | COMPRESSION_MODE2 | COMPRESSION_BW64,
    COMPRESSION_XO | COMPRESSION_MODE2 | COMPRESSION_BW64,
    COMPRESSION_CHAN1 | COMPRESSION_CHAN2 | COMPRESSION_CHAN3 | COMPRESSION_CHAN4 | COMPRESSION_MODE2 | COMPRESSION_BW64
  };

static int nvalid_compression_modes = 29;

/* this structure obtained from net-tools-1.60, interface.h */
struct user_net_device_stats {
    unsigned long long rx_packets;	/* total packets received       */
    unsigned long long tx_packets;	/* total packets transmitted    */
    unsigned long long rx_bytes;	/* total bytes received         */
    unsigned long long tx_bytes;	/* total bytes transmitted      */
    unsigned long rx_errors;	/* bad packets received         */
    unsigned long tx_errors;	/* packet transmit problems     */
    unsigned long rx_dropped;	/* no space in linux buffers    */
    unsigned long tx_dropped;	/* no space available in linux  */
    unsigned long rx_multicast;	/* multicast packets received   */
    unsigned long rx_compressed;
    unsigned long tx_compressed;
    unsigned long collisions;

    /* detailed rx_errors: */
    unsigned long rx_length_errors;
    unsigned long rx_over_errors;	/* receiver ring buff overflow  */
    unsigned long rx_crc_errors;	/* recved pkt with crc error    */
    unsigned long rx_frame_errors;	/* recv'd frame alignment error */
    unsigned long rx_fifo_errors;	/* recv'r fifo overrun          */
    unsigned long rx_missed_errors;	/* receiver missed packet     */
    /* detailed tx_errors */
    unsigned long tx_aborted_errors;
    unsigned long tx_carrier_errors;
    unsigned long tx_fifo_errors;
    unsigned long tx_heartbeat_errors;
    unsigned long tx_window_errors;
};

#define MAC_LENGTH 19
#define IP_LENGTH  16

/* linked list to hold info about each network interface */
typedef struct _network_interfaces {
  char interface_name[IFNAMSIZ];      /* the system identifier for the interface */
  char MAC[MAC_LENGTH];               /* MAC address of the interface */
  char ip_address[IP_LENGTH];         /* IP address (v4) of the interface */
  int mtu;                            /* MTU setting of the interface */
  struct user_net_device_stats stats; /* interface statistics */
  int statistics_valid;               /* are the statistics valid? */
  struct _network_interfaces *next;   /* pointer to the next available interface */
} network_interfaces;

/* linked list to hold the available output disks */
typedef struct _outputdisk {
  char diskpath[BUFFSIZE];           /* the path to the recording area of the disk */
  char disklabel[BUFFSIZE];          /* the label on the disk */
  int disknumber;                    /* the assigned disk number */
  int max_rate;                      /* the maximum data rate (in Mbps) this disk is capable of */
  int is_mounted;                    /* is the disk actually mounted? */
  int get_label;                     /* do we need to get the disk label? */
  int is_acceptable;                 /* is this disk on the user's list of selectable disks? */
  char filesystem[BUFFSIZE];         /* which filesystem is being used? */
  unsigned long long freespace;      /* the amount of free space on the disk */
  unsigned long long totalspace;     /* the total amount of disk capacity */
  struct _outputdisk *next;          /* pointer to the next disk */
  struct _outputdisk *previous;      /* pointer to the previous disk */
} outputdisk;

/* structure to hold details of a receiver */
typedef struct _receiver_details {
  int receiver_pid;                 /* the PID of the receiver process */
  char receiver_command[BUFFSIZE];  /* the command used to run the receiver */
  time_t receiver_start;            /* the time the receiver started */
  time_t receiver_end;              /* the time to kill the receiver */
  outputdisk *recordingdisk;        /* the disk used for this receiver */
} receiver_details;

/* linked list to hold details of remote recorders */
typedef struct _remoterecorder {
  char commonname[BUFFSIZE];    /* a nickname for the machine */
  char hostname[BUFFSIZE];      /* the fully qualified hostname, recorder_server will look it up */
  struct hostent *host;         /* the IP address after lookup */
  int  connectionsocket;        /* the socket we are using to connect to this host */
  int  recorderserver_port;     /* which port is recorder_server listening to on the remote host */
  int  data_communication_port; /* which port is used for recording? */
  float tcp_window_size;        /* TCP window size */
  int  evlbi_enabled;           /* an eVLBI recorder doesn't have disks, and doesn't care about our disks */
  int  udp_enabled;             /* recording using UDP/MTU size */
  int  ipd;                     /* the interpacket delay for UDP data */
  outputdisk *available_disks;  /* the disks that are available on the remote host */
  receiver_details *receiver;   /* details of a receiver running from this recorder */
  int  comms_failures;          /* the number of successive communications failures to this host */
  int  comms_wait;              /* if set, don't attempt communications */
  struct _remoterecorder *next;
} remoterecorder;

/* structure to hold the required data variables */
typedef struct _recorder {
  char  record_time[BUFFSIZE];       /* the time to record data for */
  char  record_start_date[BUFFSIZE]; /* the date to start the recording (yyyymmdd) */
  char  record_start_time[BUFFSIZE]; /* the time to start the recording (HHMMSS) */
  char  directory_name[BUFFSIZE];    /* the directory to put the files in */
  char  filename_prefix[BUFFSIZE];   /* the prefix for the output filenames */
  char  compression[BUFFSIZE];       /* the channels to record the data for */
  int   vsib_mode;                   /* the vsib mode for bit length */
  int   bandwidth;                   /* the recording bandwidth */
  outputdisk *recordingdisk;         /* a pointer to the information about the disk to record to */
  remoterecorder *targetrecorder;    /* a pointer to the information about the recorder to use (NULL for local) */
  int   n_recorders;                 /* the number of available recording disks */
  int   fringe_test;                 /* flag to specify whether this is a fringe test (1=yes,0=no) */
  char  clean_time[BUFFSIZE];        /* a character representation for the clean time, the time
				        to keep recorded data before it is deleted; if this is a
				        zero-length string, all data is kept */
  char  vsib_device[BUFFSIZE];       /* the location of the VSIB recording device */
  float filesize_or_time;            /* the size of each recorded file in blocks/seconds */
  int   filesize_is_time;            /* is the filesize_or_time variable a time (yes/no) */
  float blocksize;                   /* the size of each block in B */
  float clockrate;                   /* the rate of the VSIB clock in MHz*/
  int   auto_disk_select;             /* whether the server should automatically select the best
				        disk when starting a recording 
				        AUTODISK_DISABLED  requires a user-selected disk
				        AUTODISK_ANY       will select the best disk of all available
					AUTODISK_LOCAL     will only select from local disks
					AUTODISK_REMOTE    will only select from remote disks */

  int   mark5b_enabled;              /* recording to the Mark5B? (yes/no) */
  int   udp_enabled;                 /* Recording using UDP/MTU size */
  int   onebit_enabled;              /* 1 bit recording */
  /* the rest of the settings here are non-function settings, that is, */
  /* they only exist to be put into the header of the recorder output */
  /* files */
  int   numbits;                     /* the number of bits used to encode each sample */
  int   ipd;                         /* The interpacket delay for UDP data */
  char  encoding[BUFFSIZE];          /* how the data is encoded: AT   = offset binary 
				                                 VLBA = sign-magnitude */
  char  frequency[BUFFSIZE];         /* the lower band edge frequency of each channel in MHz,
					as a whitespace delimited list of floating point numbers */
  char  polarisation[BUFFSIZE];      /* the polarisation of each channel, as a whitespace 
					deliminated list of R or L (for Rcp and Lcp) */
  char  sideband[BUFFSIZE];          /* the sideband of each channel, as a whitespace deliminated
					list or U or L (for non-inverted or inverted spectra) */
  char  referenceant[BUFFSIZE];      /* The actual antenna position used for an "antenna" where 
					there might be some ambiguity */
} recorder;

/* linked list to account for recorder health */
typedef struct _recorderhealth {
  time_t file_time;                  /* the time the file was opened */
  char   file_name[BUFFSIZE];        /* the name of the opened file */
  int    file_block;                 /* the first block of the opened file */
  int    BIGBUF_level;               /* the amount of free BIGBUF (MB) */
  int    BIGBUF_pct;                 /* the percentage of free BIGBUF */
  int    PPS_OK;                     /* was the PPS signal received (YES/NO) */
  char   PPS_message[BUFFSIZE];      /* the PPS skipped error message */
  char   statistics[BUFFSIZE];       /* the sampler statistics of the file */
  struct _recorderhealth *next;      /* a pointer to the next file */
} recorder_health;

/* structure to hold the overall stats for the last 
   recording */
typedef struct _lastrecordinghealth {
  int valid_info;                    /* does this structure contain valid info (YES/NO) */
  time_t file_time;                  /* the time of the list opened file */
  char file_name[BUFFSIZE];          /* the name of the last opened file */
  int file_block;                    /* the last block that was opened */
  int BIGBUF_level;                  /* the last amount of free BIGBUF (MB) */
  int BIGBUF_pct;                    /* the last percentage of free BIGBUF */
  int PPS_OK;                        /* was the last PPS signal received (YES/NO) */
  char PPS_message[BUFFSIZE];        /* the last PPS skipped error message */
  int total_files;                   /* the number of files in the last recording (limit 100) */
  int total_skips;                   /* the number of PPS skips in the last total_files files */
  char statistics[BUFFSIZE];         /* the last file's sampler statistics */
  outputdisk *recorded_to_disk;      /* the disk the data was recorded to */
  char recorded_path[BUFFSIZE];      /* the path the data was recorded to */
} last_recording_health;

/* structure for a recording target */
typedef struct _rectarget {
  recorder *recorder_settings;       /* the recording settings for this target */
  char target_identifier[BUFFSIZE];  /* a way of identifying this target */
  int  is_recording;                 /* is this target currently being recorded to? */
  struct _rectarget *next;           /* the next target */
} rectarget;

/* structure for getting the status of the server */
typedef struct _status {
  char status_recorder[BUFFSIZE];    /* the status of the recorder */
  int  is_recording;                 /* has a vsib_record process been started? */
  int  in_dir;                       /* have we already changed into the output directory? */
  int  recorder_pid;                 /* the PID of the recorder process */
  char recorder_command[BUFFSIZE];   /* the command used to run the recorder */
  time_t recorder_start;             /* the time the recorder was started */
  time_t recorder_end;               /* the time the recorder is expected to end*/
  int  time_verified;                /* has the start time of the recorder been verified? */
  outputdisk *recording_to_disk;     /* the disk that is being recorded to */
  int  healthcheck_pid;              /* the PID of the health checker process */
  char status_checker[BUFFSIZE];     /* the status of the checker */
  char recording_path[BUFFSIZE];     /* the full path of the recording location */
  int  is_receiving;                 /* has a vsib_recv process been started? */
  int  receiver_pid;                 /* the PID of the receiver process */
  char receiver_command[BUFFSIZE];   /* the command used to run the receiver */
  time_t receiver_start;             /* the time the receiver started */
  time_t receiver_end;               /* the time to kill the receiver */
  char status_receiver[BUFFSIZE];    /* the status of the receiver */
  int  experiment_mode;              /* experiment mode:
					EXPERIMENT_MANUAL is manual mode
					EXPERIMENT_QUEUE  is experiment queue */
  int  execute_experiment;           /* allow automatic start for experiments? 
					AUTO_EXPERIMENT_YES  is automatic start 
				        AUTO_EXPERIMENT_NO   is manual start 
				        AUTO_EXPERIMENT_STOP is a cross between auto and manual */
  recorder *current_settings;        /* pointer to the current settings */
  recorder *recording_settings;      /* pointer to the settings of the current recording */
  char status_server[BUFFSIZE];      /* the status of the server */
  int  low_disk_action;              /* the action to take when disk space becomes critical 
				        CRITICAL_DONOTHING is do nothing
					CRITICAL_STOP      stops the recorder
					CRITICAL_SWITCH    switches the recorder to another disk */
  time_t server_start;               /* the start time of the server */
  int  last_block;                   /* the last recorded start block */
  char last_file[BUFFSIZE];          /* the last recorded file */
  char last_pps[BUFFSIZE];           /* the status of the last PPS signal */
  long unsigned int last_bigbuf;     /* the amount of BIGBUF free */
  int  last_bigbuf_pct;              /* percentage of free BIGBUF */
  recorder_health *recstatus;        /* the health of the recorder */
  last_recording_health lastrec;     /* the overall health of the last recording */
  int  disk_warning_level;           /* what warning level have we reached 
				        WARNING_NONE     = no warnings yet issued 
				        WARNING_ONE      = low space first warning issued 
				        WARNING_TWO      = low space second warning issued 
				        WARNING_CRITICAL = critically low space warning issued */
  int  rounded_start;                /* whether to restrict recorder start to filesize boundaries 
				        ROUNDSTART_YES   = always start recording on filesize boundaries 
				        ROUNDSTART_NO    = start immediately when asked to start */
  network_interfaces *interfaces;    /* the network interfaces available */
  FILE *experiment_report;           /* a file to record the filenames recorded in the latest experiment */
  remoterecorder *remote_recorders;  /* the remote recorders that are available */
  int update_remote_disks;           /* a flag to specify whether to update the remote host (YES/NO) */
  rectarget *target_list;            /* the list of available recording targets */
} server_status;

/* linked list for experiment queuing */
typedef struct _experiment {
  char  experiment_id[BUFFSIZE];     /* the experiment identifier */
  time_t start_time;                 /* the start time for the experiment */
  time_t end_time;                   /* the end time */
  int   started;                     /* has the experiment been started? (YES/NO) */
  recorder *record_settings;         /* the experiment configuration */
  struct _experiment *next;          /* pointer to the next experiment */
} experiment;

/* linked list for registering clients for receiving messages */
typedef struct _registeredclients {
  char ip_address[BUFFSIZE];         /* the ip address of the client */
  int  port;                         /* the port it is listening for messages on */
  struct _registeredclients *next;   /* a pointer to the next client */
} registered_clients;


/* linked list for recorder error messages */
typedef struct _recordererrors {
  char recorder_error_message[BUFFSIZE]; /* the error message produced by vsib_record */
  char broadcast_message[BUFFSIZE];      /* the message to send to clients if error encountered */
  struct _recordererrors *next;
} recordererrors;


/* the system commands */
#define vsib_record_command    "/home/vlbi/bin/vsib_record"
#define vsib_checker_command   "/home/vlbi/bin/vsib_checker"
#define perl_command           "/usr/bin/perl"
#define perl_options           "-w"
#define disk_clean_command     "/home/vlbi/bin/diskclean.pl"
#define health_checker_command "/home/vlbi/bin/recorder_health_checker"
#define hdparm_command         "/sbin/hdparm"
#define serial_command         "/home/vlbi/bin/get_disk_serials"
#define recv_command           "/home/vlbi/bin/vsib_recv"

/* directory locations */
#define experiment_location  "/home/vlbi/experiment_profiles"
#define default_directory    "/home/vlbi"
#define log_location         "/home/vlbi/recorder/logs"
#define tmp_directory        "/tmp"
#define disks_config_file    "/home/vlbi/recorder/recorder_disks.conf"

/* default parameters */
#define DEFAULT_DEVICE       "/dev/vsib"
#define DEFAULT_FILESIZE     10
#define DEFAULT_FILETIME     YES
#define DEFAULT_BLOCKSIZE    32000
#define DEFAULT_CLOCKRATE    32
#define DEFAULT_EVLBI        NO
#define DEFAULT_MARK5B       NO
#define DEFAULT_UDP          0
#define DEFAULT_NUMBITS      -1
#define DEFAULT_IPD          0
#define DEFAULT_ENCODING     "\0"
#define DEFAULT_FREQUENCY    "\0"
#define DEFAULT_POLARISATION "\0"
#define DEFAULT_SIDEBAND     "\0"
#define DEFAULT_REFERENCEANT "\0"

/* This is a list of interface name prefixes which are `bad' in the sense
 * that they don't refer to interfaces of external type on which we are
 * likely to want to listen. We also compare candidate interfaces to lo. */
static char *bad_interface_names[] = {
  "lo:",
  "lo",
  "stf",     /* pseudo-device 6to4 tunnel interface */
  "gif",     /* psuedo-device generic tunnel interface */
  "dummy",
  "vmnet",
  NULL        /* last entry must be NULL */
};

/* function definitions */
void PrintLog(char *mess);
void Die(char *mess, int sock);
void PrintStatus(char *mess);
void DebugStatus(char *mess);
void IssueWarning(char *mess);
void BroadcastMessage(char *mess);
void HandleClient(int sock,recorder *settings);
int health_handler(char *data,char *failure,int type);
int command_handler(char *data,char *failure,recorder *settings);
int receiver_control(int action,char *commonname,recorder *settings,char *failure);
int directory_default(char *failure);
int experiment_control(int action,char *argument,char *failure);
int list_experiments(char *pattern,char *names,char ***locations,int *nfiles,char *failure);
static int one(const struct dirent *unused);
void settings_string(recorder *settings,char *output);
int status_control(int mode,recorder *settings,char *failure);
static int is_bad_interface_name(char *i);
int interface_config(char *failure);
int check_control(int mode,recorder *settings,char *failure,char check_args[2][BUFFSIZE]);
int recorder_control(int command,recorder *settings,char *failure);
int health_control(int command,char *failure);
int data_handler(char *data,char *failure,recorder *settings);
int check_time(int year,int month,int date,int hour,int minute,int second);
int check_date(int year,int month,int date);
int xmody(int x,int y);
void send_confirmation(int errorcode,char *errormessage,int sock);
int message_handler(char *message,char *failure);
/* this is where main is */
void server_stop(int sig);
int LoadSettings(recorder *settings,int RESET,char *failmsg);
int check_disk_mount(char *diskpath,char *filesystem);
int GetSettings(recorder *settings,int RESET,char *failmsg);
int LoadServerSettings(recorder *settings,char *failure);
int LoadExperiment(char *profilename,char *failure,int recovermode);
int SaveExperiment(char *expid,char *failure);
int LoadRecorderSettings(recorder *settings,char *failmsg);
int WriteRecorderSettings(recorder *settings,char *failure);
int checkrunning(int pid,char *pname);
void assign_time(int year,int month,int day,int hour,int minute,int second,time_t *t_assigned);
void thetimenow(char *time_string);
void socket_options(int sock,int call_flag);
void RemoveRemote(remoterecorder *badrecorder,int removeflag);
int AddRemote(char *message,remoterecorder *newrecorder);
int PushRemote(char *message,remoterecorder *sendrecorder);
void PushStatus(char *message);
void RegisterClient(char *ipaddress,int port);
void RemoveClient(char *ip_address,int port_number);
void initialise_status(recorder *settings);
int StartExperiment(char *failure);
int ExperimentReady(int action,char *failure);
void UpdateStatusSignal(int sig);
int UpdateStatus(char *failure);
void if_readlist_proc(void);
int get_dev_fields(char *bp,network_interfaces *ife,int procnetdev_vsn);
char *get_name(char *name, char *p);
int procnetdev_version(char *buf);
int RecorderErrors(char *returned_lines);
int timeinseconds(char *time_expression);
void timeasstring(time_t that_time,char *time_string);
int free_space(char *diskpath,unsigned long long *freespace,unsigned long long *totalspace,char *failure);
int recordingdatarate(recorder *settings);
int recordingfreetime(unsigned long long freespace,int datarate);
void time_difference(time_t first_time,time_t second_time,char *string_representation);
int experiment_add(experiment *newExperiment,char *failure);
int swapdisk(int flag,recorder *settings,char *failure);
void bestdisk(recorder *settings,outputdisk **best_disk,
	      remoterecorder **best_recorder,outputdisk *excludedisk);
int disk_critical_action(char *failure);
void fuzzy_time(time_t time_in_seconds,char *representation);
int get_drive_serial(char *serial_numbers,char *filesystem,char *failure,int flag);
void conjugate_host(char *normal,char *conjugate);
void copy_recorder(recorder *in,recorder *out);
void copy_hostent(struct hostent *dest,struct hostent *src);
void free_hostent(struct hostent *dest);
int copy_recordertarget(int flag,char *target_id,recorder *settings,char *failure);
