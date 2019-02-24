#!/usr/bin/perl

# This script will install the recorder server program after
# getting the answers to a few questions.

# check whether the recorder server could already be running
if (-e "/etc/init.d/recorder_server"){
    system "/etc/init.d/recorder_server stop";
}

# print the welcome message
print "recorder_server_threads installer\n";
print "This is release 1 of the threaded server.\n";
$installer_source_version=1.1;
$installer_header_version=1.1;
print "\nThis script will:\n";
print " 1. Compile and install recorder_server_threads v$installer_source_version\n";
print " 2. Configure recorder_disks.conf for your machine.\n";
print " 3. Install the server as a system service (optional).\n";

print "\n== YOU SHOULD RUN THIS INSTALL SCRIPT AS THE USER WHICH ==\n";
print   "==  USUALLY RUNS THE vsib_record COMMAND, BUT YOU WILL  ==\n";
print   "==    BE REQUIRED TO SUPPLY THE ROOT PASSWORD LATER     ==\n";

print "\nPlease answer the following questions.\n";

# the location of the machine
# we can figure this out from the name of the machine
open(MACHINE,"-|")||exec "uname -n";
while(<MACHINE>){
    chop;
    $machine_name=$_;
}
close(MACHINE);
# the recorder machines are generally called
# ##vsi where ## is the abbreviation of the station
if ($machine_name=~/..vsi(.*)/){
    $location=$machine_name;
    $location=~s/(..)vsi(.*)/$1/;
} else {
    $location="ts";
}

# define some standard locations
@codes = ("43","45","ho","cd","mp","at","pk","ts");
@locations = ("DSS43","DSS45","Hobart","Ceduna","Mopra","ATCA","Parkes","Test");

print "\nThe location of your machine:\n";
for ($i=0;$i<=$#codes;$i++){
    print "  $codes[$i] = $locations[$i]\n";
}
$my_location=$location;
$read_location=&ask_question(" Enter the two digit code corresponding to your location",$my_location);
if ($read_location eq "un"){
    print "  Location chosen: unknown\n";
} else {
    $fl=0;
    for ($i=0;$i<=$#codes;$i++){
	if (lc($read_location) eq $codes[$i]){
	    print "  Location chosen: $locations[$i]\n";
	    $fl=1;
	}
    }
    if ($fl==0){
	print "  Location chosen: $read_location\n";
    }
}

# the available recording disks
# we have changed location for this config file as of v1.47 to
# /home/vlbi/recorder/recorder_disks.conf, so we check if that
# exists first, as it should be more up to date. if it doesn't
# exists, we check /etc/recorder_disks.conf, and use it.
# of course, /home/vlbi depends on running it as vlbi, which we
# check for now
$iamuser=&who_am_i();
if ($iamuser eq "root"){
    $got_root=1;
}
if (-e "/home/$iamuser/recorder/recorder_disks.conf"){
    open(RECDISKS,"/home/$iamuser/recorder/recorder_disks.conf");
    while(<RECDISKS>){
	chop;
	@els=split(/\s+/);
	if (($els[0]!~/^\#/)&&($els[0] ne "STATION")){
	    push @disk_name,$els[0];
	    push @disk_speed,$els[1];
	}
    }
    close(RECDISKS);
} elsif (-e "/etc/recorder_disks.conf"){
    open(RECDISKS,"/etc/recorder_disks.conf");
    while(<RECDISKS>){
	chop;
	@els=split(/\s+/);
	if (($els[0]!~/^\#/)&&($els[0] ne "STATION")){
	    push @disk_name,$els[0];
	    push @disk_speed,$els[1];
	}
    }
    close(RECDISKS);
}

# go through /etc/fstab and look for probable entries
# these should be the standard disks
# /data/internal
# /data/removable
# /data/xraid*
open(FSTAB,"/etc/fstab");
while(<FSTAB>){
    chop;
    @els=split(/\s+/);
    $temp_disk="";
    $temp_speed="";
    if ($els[0]!~/\#/){
	if ($els[1]=~/\/data\/internal/){
	    $temp_disk=$els[1];
	    $temp_speed="256";
	} elsif ($els[1]=~/\/data\/removable/){
	    $temp_disk=$els[1];
	    $temp_speed="512";
	} elsif ($els[1]=~/\/data\/xraid/){
	    $temp_disk=$els[1];
	    $temp_speed="512";
	}
    }
    if ($temp_disk ne ""){
	$fd=0;
	for ($i=0;$i<=$#disk_name;$i++){
	    if ($temp_disk eq $disk_name[$i]){
		$fd=1;
		last;
	    }
	}
	if ($fd==0){
	    push @disk_name,$temp_disk;
	    push @disk_speed,$temp_speed;
	}
    }
}
close(FSTAB);

# now ask the user what they think about it
while (1){
    print "\nThe available recording disks:\n";
    $availdisks=0;
    $maxlen=0;
    $mindisk=1000;
    $maxdisk=0;
    for ($i=0;$i<=$#disk_name;$i++){
	if (length($disk_name[$i])>$maxlen){
	    $maxlen=length($disk_name[$i]);
	}
    }
    print "   N  Disk path";
    for ($i=0;$i<=$maxlen-9;$i++){
	print " ";
    }
    print "Speed (Mb/s)\n";
    for ($i=0;$i<=$#disk_name;$i++){
	if ($disk_name[$i] ne ""){
	    printf ("  %2d  %s",$i+1,$disk_name[$i]);
	    for ($j=0;$j<=$maxlen-length($disk_name[$i]);$j++){
		print " ";
	    }
	    print "$disk_speed[$i]\n";
	    $availdisks++;
	    if (($i+1)<$mindisk){
		$mindisk=$i+1;
	    }
	    if (($i+1)>$maxdisk){
		$maxdisk=$i+1;
	    }
	}
    }
    if ($availdisks==0){
	print "   No disks available\n";
    }
    print " Do you wish to (a)dd another disk,\n";
    print "                (d)elete a disk,\n";
    print "                (c)hange a disk's speed\n";
    print "                (s)ave this table and continue\n";
    $defchoice="s";
    $read_choice=&ask_question(" Enter choice","a","d","c","s",$defchoice);
    if ($read_choice eq "a"){
	print " Adding disk:\n";
	print "  disk path  = ";
	chop($new_path=<STDIN>);
	print "  disk speed = ";
	chop($new_speed=<STDIN>);
	if (($new_path ne "")&&($new_speed ne "")){
	    print "  added $new_path\n";
	    push @disk_name,$new_path;
	    push @disk_speed,$new_speed;
	}
    } elsif ($read_choice eq "d"){
	if ($availdisks>0){
	    print " Delete which disk ($mindisk-$maxdisk): ";
	    chop($deldisk=<STDIN>);
	    if (($deldisk>=$mindisk)&&($deldisk<=$maxdisk)){
		print "  $disk_name[$deldisk-1] deleted.\n";
		$disk_name[$deldisk-1]="";
		$disk_speed[$deldisk-1]="";
	    }
	}
    } elsif ($read_choice eq "c"){
	if ($availdisks>0){
	    print " Change speed of which disk ($mindisk-$maxdisk): ";
	    chop($changedisk=<STDIN>);
	    if (($changedisk>=$mindisk)&&($changedisk<=$maxdisk)){
		print " Enter new speed for $disk_name[$changedisk-1]: ";
		chop($new_speed=<STDIN>);
		if ($new_speed ne ""){
		    print "  speed of $disk_name[$changedisk-1] now $new_speed\n";
		    $disk_speed[$changedisk-1]=$new_speed;
		}
	    }
	}
    } else {
	if ($availdisks==0){
	    print " THERE MUST BE AT LEAST ONE DISK!\n";
	} else {
	    last;
	}
    }
}

$disk_conf_location="/home/$iamuser/recorder/recorder_disks.conf";
$written=0;
while($written==0){
    print "\nWill write configuration to $disk_conf_location\n";
    $yesno=&ask_question(" Is this OK","y","n","y");
    if (lc($yesno) eq "n"){
	$tl=&ask_question_cs(" Where should the configuration file be saved");
	if ($tl ne ""){
	    $disk_conf_location=$tl;
	    $written=1;
	}
    } elsif (lc($yesno) eq "y"){
	$written=1;
    }
}

# make the directory if necessary
if ($disk_conf_location=~/(.*)\/recorder_disks\.conf/){
    $disk_conf_location=~s/(.*)\/recorder_disks\.conf/$1/;
}
if ($disk_conf_location=~/\/$/){
    $disk_conf_location=~s/\/$//;
}
&make_directory($disk_conf_location,$iamuser);

open(DISKS,">recorder_disks.conf");
$lcode=ucfirst($read_location);
print DISKS "STATION = $lcode\n";
for ($i=0;$i<=$#disk_name;$i++){
    if ($disk_name[$i] ne ""){
	print DISKS "$disk_name[$i]  $disk_speed[$i]\n";
    }
}
close(DISKS);
print "Disk table saved.\n";

# the install path
# defaults to /home/vlbi/bin
print "\nInstallation path:\n";
$defaultpath="/home/$iamuser/bin";
$installpath=&ask_question_cs(" The recorder server programs will be installed in",
			   $defaultpath);
if ($installpath=~/\/$/){
    $installpath=~s/\/$//;
}
if (!-e $installpath){
    $yesno=&ask_question("  Directory $installpath does not exist; create it now","y","n","y");
    if (lc($yesno) ne "n"){
	&make_directory($installpath,$iamuser);
    }
}
	
# check if there is already a recorder_server in the install path
if (-e "$installpath/recorder_server_threads"){
    $continue_install=0;
    open(RECVERSION,"-|")||exec "$installpath/recorder_server_threads -v";
    while(<RECVERSION>){
	chop;
	@els=split(/\s+/);
	if ($els[0] eq "source"){
	    $old_source_version=$els[2];
	} elsif ($els[0] eq "header"){
	    $old_header_version=$els[2];
	}
    }
    close(RECVERSION);
    if (($installer_source_version<$old_source_version)||
	($installer_header_version<$old_header_version)){
	print " The currently installed recorder_server_threads is v$old_source_version/$old_header_version,\n";
	print " which is newer than the version to be installed (v$installer_source_version/$installer_header_version).\n";
	$yesno=&ask_question(" Are you sure you wish to install this older version","y","n","n");
	if (lc($yesno) eq "y"){
	    $continue_install=1;
	} else {
	    $continue_install=0;
	}
    } else {
	print " The currently installed recorder_server_threads is v$old_source_version/$old_header_version,\n";
	print " which is older than the version to be installed (v$installer_source_version/$installer_header_version).\n";
	$yesno=&ask_question(" Do you wish to install this newer version","y","n","y");
	if (lc($yesno) eq "n"){
	    $continue_install=0;
	} else {
	    $continue_install=1;
	}
    }
} else {
    $continue_install=1;
}
if ($continue_install==1){
    print " The recorder_server_threads (v$installer_source_version/$installer_header_version) will be installed in ";
    print " $installpath\n";
} else {
    print "\nAborting install.\n";
    exit;
}

# change the header file to look in the right locations for
# the programs we are about to install
# keep an original copy of the header file, or restore the
# original if it exists already
if (-e "recorder_server_threads.h.orig"){
    $sys_command="cp -f recorder_server_threads.h.orig recorder_server_threads.h";
    system $sys_command;
} else {
    $sys_command="cp -f recorder_server_threads.h recorder_server_threads.h.orig";
    system $sys_command;
}
$sys_command="sed \'s|\"/home/vlbi/bin/recorder_health_checker\"|\"$installpath/recorder_health_checker\"|\' < recorder_server_threads.h > new_recorder_server_threads.h";
system $sys_command;
$move_command="mv -f new_recorder_server_threads.h recorder_server_threads.h";
system $move_command;
$sys_command="sed \'s|\"/home/vlbi/bin/get_disk_serials\"|\"$installpath/get_disk_serials\"|\' < recorder_server_threads.h > new_recorder_server_threads.h";
system $sys_command;
system $move_command;
$sys_command="sed \'s|\"/home/vlbi/recorder/recorder_disks.conf\"|\"$disk_conf_location/recorder_disks.conf\"|\' < recorder_server_threads.h > new_recorder_server_threads.h";
system $sys_command;
system $move_command;

# ask which user should be used to run the server as
print "\n";
$default_run_as_user="$iamuser";
if ($installpath=~/\/home\/([^\/]*)\/(.*)/){
    # somebody's home directory
    $default_run_as_user=$installpath;
    $default_run_as_user=~s/\/home\/([^\/]*)\/(.*)/$1/;
}
$run_as_user="";
$user_cont=0;
while($user_cont==0){
    $run_as_user=&ask_question_cs(" Which user account should be used to run the recorder server",$default_run_as_user);
    open(USERS,"/etc/passwd");
    $fu=0;
    while(<USERS>){
	chop;
	@els=split(/\:/);
	if ($els[0] eq $run_as_user){
	    $fu=1;
	}
    }
    close(USERS);
    if ($fu==1){
	$user_cont=1;
    }
}

# default locations
print "\n";
print " Use default locations for experiment profiles (/home/$run_as_user/experiment_profiles),\n";
print " logs (/home/$run_as_user/recorder/logs),";
$yesno=&ask_question(" and temporary storage (/tmp)","y","n","y");
if (lc($yesno) eq "n"){
    $continue_install=0;
} else {
    $experiment_location="/home/$run_as_user/experiment_profiles";
    $log_location="/home/$run_as_user/recorder/logs";
    $tmp_location="/tmp";
    $continue_install=1;
}
while($continue_install==0){
    $experiment_location="";
    $log_location="";
    $tmp_location="";
    $experiment_location=&ask_question_cs(" Full path to where experiment profiles are to be stored");
    $log_location=&ask_question_cs(" Full path to where logs will be kept");
    $tmp_location=&ask_question_cs(" Full path for temporary storage");
    if (($experiment_location ne "")&&
	($log_location ne "")&&
	($tmp_location ne "")){
	$continue_install=1;
    }
}
# make these directories if necessary
&make_directory($experiment_location,$run_as_user);
&make_directory($log_location,$run_as_user);
&make_directory($tmp_location,$run_as_user);

$default_location="/home/$run_as_user";
$sys_command="sed \'s|\"/home/vlbi/experiment_profiles\"|\"$experiment_location\"|\' < recorder_server_threads.h > new_recorder_server_threads.h";
system $sys_command;
$move_command="mv -f new_recorder_server_threads.h recorder_server_threads.h";
system $move_command;
$sys_command="sed \'s|\"/home/vlbi/recorder/logs\"|\"$log_location\"|\' < recorder_server_threads.h > new_recorder_server_threads.h";
system $sys_command;
system $move_command;
$sys_command="sed \'s|\"/home/vlbi\"|\"$default_location\"|\' < recorder_server_threads.h > new_recorder_server_threads.h";
system $sys_command;
system $move_command;
$sys_command="sed \'s|\"/tmp\"|\"$tmp_location\"|\' < recorder_server_threads.h > new_recorder_server_threads.h";
system $sys_command;
system $move_command;

# now figure out where vsib_record and those programs are
$location=&super_command("which vsib_record",$run_as_user);
if (($location=~/(.*)no\svsib_record(.*)/)||
    ($location eq "")){
    # couldn't find vsib_record
    print " Unable to find vsib_record in the path of $run_as_user!\n";
    $loc_cont=0;
    while($loc_cont==0){
	$tmp=&ask_question_cs(" Enter the full path to vsib_record");
	if ((-e $tmp)&&(!-d $tmp)&&(-x $tmp)){
	    $location=$tmp;
	    $loc_cont=0;
	} else {
	    print "  $tmp is not an executable!\n";
	}
    }
}
$check_path=$location;
$check_path=~s/(.*)\/(.*)/$1/; # relying on regexp greediness here
$vsibrecord_location=$location;
print "found vsib_record at $vsibrecord_location\n";
$location="/home/vlbi/chris/vsib-multi/vsib_record";
if (!-e $location){
    # couldn't find the threaded vsib_record
    print " Unable to find threaded vsib_record at $location!\n";
    $loc_cont=0;
    while($loc_cont==0){
	$tmp=&ask_question_cs(" Enter the full path to the threaded vsib_record");
	if ((-e $tmp)&&(!-d $tmp)&&(-x $tmp)){
	    $location=$tmp;
	    $loc_cont=0;
	} else {
	    print "  $tmp is not an executable!\n";
	}
    }
}
$thread_vsibrecord_location=$location;
print "found threaded vsib_record at $location\n";
$location=&super_command("which vsib_recv",$run_as_user);
if (($location=~/(.*)no\svsib_recv(.*)/)||
    ($location eq "")){
    $location_check="$check_path/vsib_recv";
    if ((-e $location_check)&&(!-d $location_check)&&
	(-x $location_check)){
	$location=$location_check;
    } else {
	# couldn't find vsib_recv
	print " Unable to find vsib_recv in the path of $run_as_user!\n";
	$loc_cont=0;
	while($loc_cont==0){
	    $tmp=&ask_question_cs(" Enter the full path to vsib_recv");
	    if ((-e $tmp)&&(!-d $tmp)&&(-x $tmp)){
		$location=$tmp;
		$loc_cont=1;
	    } else {
		print "  $tmp is not an executable!\n";
	    }
	}
    }
}
$vsibrecv_location=$location;
print "found vsib_recv at $vsibrecv_location\n";
$location=&super_command("which perl",$run_as_user);
if (($location=~/(.*)no\sperl(.*)/)||
    ($location eq "")){
    $location_check="$check_path/perl";
    if ((-e $location_check)&&(!-d $location_check)&&
	(-x $location_check)){
	$location=$location_check;
    } else {
	# couldn't find perl
	print " Unable to find perl in the path of $run_as_user!\n";
	$loc_cont=0;
	while($loc_cont==0){
	    $tmp=&ask_question_cs(" Enter the full path to perl");
	    if ((-e $tmp)&&(!-d $tmp)&&(-x $tmp)){
		$location=$tmp;
		$loc_cont=1;
	    } else {
		print "  $tmp is not an executable!\n";
	    }
	}
    }
}	
$perl_location=$location;
print "found perl at $perl_location\n";

# further change the header file
$sys_command="sed \'s|\"/home/vlbi/bin/vsib_record\"|\"$vsibrecord_location\"|\' < recorder_server_threads.h > new_recorder_server_threads.h";
system $sys_command;
system $move_command;
$sys_command="sed \'s|\"/home/vlbi/chris/vsib-multi/vsib_record\"|\"$thread_vsibrecord_location\"|\' < recorder_server_threads.h > new_recorder_server_threads.h";
system $sys_command;
system $move_command;
$sys_command="sed \'s|\"/usr/bin/perl\"|\"$perl_location\"|\' < recorder_server_threads.h > new_recorder_server_threads.h";
system $sys_command;
system $move_command;
$sys_command="sed \'s|\"/home/vlbi/bin/vsib_recv\"|\"$vsibrecv_location\"|\' < recorder_server_threads.h > new_recorder_server_threads.h";
system $sys_command;
system $move_command;

# compile the recorder_server_threads code
print "\nNow compiling the server code...\n";
open(COMPILE,"-|")||exec "make";
$errors=0;
while(<COMPILE>){
    chop;
    $line=$_;
    print "$line\n";
    if ($line=~/(.*)\serror\:\s/){
	$errors=1;
    }
}
close(COMPILE);
if ($errors==1){
    print "\nThe server code did not successfully compile!\n";
    print "Please send a bug report with the error messages above to\n";
    print "Jamie.Stevens@utas.edu.au\n";
    exit;
}

print "\nServer code compiled successfully.\n";

# copy the binary files to the installation path
print "\nNow copying server binaries...\n";
&copy_files("recorder_server_threads","recorder_health_checker",$installpath,$run_as_user);
&copy_files("recorder_disks.conf",$disk_conf_location,$run_as_user);
print "\n We now need to make get_disk_serials run as root, so the server\n";
print " can get the serial numbers of the recording disks.\n";
print "\n = assembling shell script of commands to run as root =\n";
open(ROC,">root_commands.sh");
print "cp get_disk_serials $installpath/ && chmod +s $installpath/get_disk_serials\n";
print ROC "cp get_disk_serials $installpath/ && chmod +s $installpath/get_disk_serials\n";
#&super_command("cp get_disk_serials $installpath/ && chmod +s $installpath/get_disk_serials");
print "Copy complete.\n";

# install a script which will start the server automatically on
# system start-up
$yesno=&ask_question("\nDo you want this server installed as a service","y","n","y");
if (lc($yesno) eq "n"){
    print "To run the server, type recorder_server_threads at the command line.\n";
} else {
    open(SERVICE,">recorder_server_init.d");
    print SERVICE "#!/bin/sh\n";
    print SERVICE "SERVER_BIN=$installpath/recorder_server_threads\n";
    print SERVICE "DEFAULT_USER=$run_as_user\n";
    print SERVICE "CONFIG_FILE=$disk_conf_location/recorder_disks.conf";
    close(SERVICE);
    system "cat init_recorder_server >> recorder_server_init.d";
    print "cp recorder_server_init.d /etc/init.d/recorder_server && chmod 755 /etc/init.d/recorder_server\n";
    print ROC "cp recorder_server_init.d /etc/init.d/recorder_server && chmod 755 /etc/init.d/recorder_server\n";
#    &super_command("cp recorder_server_init.d /etc/init.d/recorder_server && chmod 755 /etc/init.d/recorder_server");
    $yesno=&ask_question("\nDo you want this service to be started on system boot","y","n","y");
    if (lc($yesno) ne "n"){
	print "/usr/sbin/update-rc.d recorder_server defaults\n";
	print ROC "/usr/sbin/update-rc.d recorder_server defaults\n";
#	&super_command("/usr/sbin/update-rc.d recorder_server defaults");
	print " The recorder server will start on bootup.\n";
    }
    print "To start the server service, type /etc/init.d/recorder_server start\n";
    print "     and to stop the server, type /etc/init.d/recorder_server stop\n";
}
close(ROC);

# now execute the root shell script
print "\n Now executing the actions needing root permissions.\n";
&super_command("/bin/sh root_commands.sh");

# that's it, the recorder is installed
print "\nInstallation is complete.\n";


# our subroutines
sub ask_question {
    if ($#_<0){
	return;
    } elsif ($#_==0){
	$question=$_[0];
	$default="";
	$use_options=0;
    } elsif ($#_==1){
	$question=$_[0];
	$default=$_[1];
	$use_options=0;
    } else {
	$question=$_[0];
	while($#options>=0){
	    pop @options;
	}
	for ($i=1;$i<$#_;$i++){
	    push @options,$_[$i];
	}
	$default=$_[$#_];
	$use_options=1;
    }

    $answered=0;
    while($answered==0){
	print "$question";
	if ($default ne ""){
	    if ($use_options==1){
		print " (";
		for ($i=0;$i<=$#options;$i++){
		    if ($options[$i] eq $default){
			$out=uc($options[$i]);
			print "$out";
		    } else {
			print "$options[$i]";
		    }
		    if ($i!=$#options){
			print "/";
		    }
		}
		print ")";
	    } elsif ($use_options==0){
		print " [$default]";
	    }
	}
	print ": ";
	
	chop($answer=<STDIN>);
	if ($answer eq ""){
	    if ($default ne ""){
		$answer=$default;
		$answered=1;
	    }
	}
	if ($use_options==1){
	    for ($i=0;$i<=$#options;$i++){
		if (lc($answer) eq lc($options[$i])){
		    $answered=1;
		}
	    }
	} elsif ($answer ne ""){
	    $answered=1;
	}
    }
    lc($answer);

}

sub ask_question_cs {
    if ($#_<0){
	return;
    } elsif ($#_==0){
	$question=$_[0];
	$default="";
	$use_options=0;
    } elsif ($#_==1){
	$question=$_[0];
	$default=$_[1];
	$use_options=0;
    } else {
	$question=$_[0];
	while($#options>=0){
	    pop @options;
	}
	for ($i=1;$i<$#_;$i++){
	    push @options,$_[$i];
	}
	$default=$_[$#_];
	$use_options=1;
    }

    $answered=0;
    while($answered==0){
	print "$question";
	if ($default ne ""){
	    if ($use_options==1){
		print " (";
		for ($i=0;$i<=$#options;$i++){
		    if ($options[$i] eq $default){
			$out=uc($options[$i]);
			print "$out";
		    } else {
			print "$options[$i]";
		    }
		    if ($i!=$#options){
			print "/";
		    }
		}
		print ")";
	    } elsif ($use_options==0){
		print " [$default]";
	    }
	}
	print ": ";
	
	chop($answer=<STDIN>);
	if ($answer eq ""){
	    if ($default ne ""){
		$answer=$default;
		$answered=1;
	    }
	}
	if ($use_options==1){
	    for ($i=0;$i<=$#options;$i++){
		if (lc($answer) eq lc($options[$i])){
		    $answered=1;
		}
	    }
	} elsif ($answer ne ""){
	    $answered=1;
	}
    }
    $answer;

}

sub make_directory {
    if ($#_<0){
	return;
    } elsif ($#_==0){
	$full_path=$_[0];
	$owner="";
    } elsif ($#_==1){
	$full_path=$_[0];
	$owner=$_[1];
    } else {
	return;
    }
    $made_something=0;
    @dels=split(/\//,$full_path);
    $made_so_far="";
    for (my $p=1;$p<=$#dels;$p++){
	$to_make="$made_so_far/$dels[$p]";
	if (-d $to_make){
	    $made_so_far=$to_make;
	} else {
	    print "Making directory $to_make\n";
	    if ($owner eq ""){
		$make_command="mkdir $to_make && chown $owner $to_make";
	    } else {
		$make_command="mkdir $to_make";
	    }
	    system $make_command;
	    if (-d $to_make){
		$made_so_far=$to_make;
		$made_something=1;
	    } else {
		print " Could not make directory $to_make!\n";
		$noyes=&ask_question(" Should I create $to_make as root","y","n","y");
		if ($noyes eq "y"){
		    print "\n YOU WILL NOW NEED TO SUPPLY THE ROOT PASSWORD\n";
		    if ($owner eq ""){
			$make_command="mkdir $to_make && chown $owner $to_make";
		    } else {
			$make_command="mkdir $to_make";
		    }
		    super_command($make_command);
		    if (-d $to_make){
			$made_so_far=$to_make;
			$made_something=1;
		    } else {
			print " Still could not make directory $to_make!\n";
			print " Please check permissions of $made_so_far. Aborting!\n";
			exit;
		    }
		} else {
		    print " Please check permissions of $made_so_far. Aborting!\n";
		    exit;
		}
	    }
	}
    }
    if ($made_something==1){
	print " Directory $full_path created.\n";
    } else {
	print " Directory $full_path already exists.\n";
    }
}

sub super_command {
    if ($#_<0){
	return;
    } elsif ($#_==0){
	$issue_command=$_[0];
	$run_as_root=1;
	$ra_user="";
    } elsif ($#_==1){
	$issue_command=$_[0];
	$run_as_root=0;
	$ra_user=$_[1];
    } else {
	return;
    }
    $which_user=&who_am_i();
    if ($run_as_root==1){
	if ($which_user ne "root"){
	    print "\n == YOU WILL NOW NEED TO SUPPLY THE ROOT PASSWORD ==\n";
	}
	$su_command="su -c '$issue_command'";
    } else {
	if (($which_user ne "root")&&($which_user ne $ra_user)){
	    print "\n == YOU WILL NOW NEED TO SUPPLY THE PASSWORD FOR $ra_user ==\n";
	}
	$su_command="su -c '$issue_command' - $ra_user";
    }
#    system $su_command;
    if ($which_user ne $ra_user){
	print "Executing command [ $su_command ]\n";
	chop($result=qx{$su_command});
    } else {
	print "Executing command [ $issue_command ]\n";
	chop($result=qx{$issue_command});
    }
    $result;
}

sub who_am_i {
    my $this_user;
    open(WHOAMI,"-|")||exec "whoami";
    while(<WHOAMI>){
	chop;
	$this_user=$_;
    }
    close(WHOAMI);
    $this_user;
}

sub copy_files {
    if ($#_<0){
	return;
    } else {
	while ($#cfiles>=0){
	    pop @cfiles;
	}
	for (my $p=0;$p<=$#_-2;$p++){
	    push @cfiles,$_[$p];
	}
	$cdest=$_[$#_-1];
	$cuser=$_[$#_];
    }
    if ($cdest=~/\/$/){
	$cdest=~s/\/$//;
    }
    $ccmd="cp";
    for (my $p=0;$p<=$#cfiles;$p++){
	$tmp=$ccmd;
	$ccmd="$tmp $cfiles[$p]";
    }
    $tmp=$ccmd;
    $ccmd="$tmp $cdest/";
    $cres=&super_command($ccmd,$cuser);
    if ($cres=~/(.*)Permission\sdenied(.*)/){
	print " Unable to copy to $cdest as $cuser.\n";
	$noyes=&ask_question(" Should I retry as root","y","n","y");
	if ($noyes eq "y"){
	    $tmp=$ccmd;
	    $ccmd="$tmp && chown $cuser";
	    for (my $p=0;$p<=$#cfiles;$p++){
		$tmp=$ccmd;
		$ccmd="$tmp $cdest/$cfiles[$p]";
	    }
	    $cres=&super_command($ccmd);
	} else {
	    print " Copy failed! Aborting.\n";
	    exit;
	}
    }
}
