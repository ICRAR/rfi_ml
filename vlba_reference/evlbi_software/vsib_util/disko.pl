#!/usr/bin/perl -w  -d
use strict;
use Carp;

my $progname = "Disko";
my $progvers = "1.22";

#######################################################################
###  Disko: A GUI for vsib_record
###
###  History:
###   16-Sep_2005 : v1.0  : Initial release    - Jim Lovell
###   10-May_2006 : v1.1  : Tweak for Tk.pm AIP change
###                         Check for vc* fringe checks - CJP
###   14-May_2006 : v1.2  : Added 64 MHz - JEJL/CJP
###   14-May_2006 : v1.21 : Added channs 1,2 & 5,6 mode
###   24-Jan_2007 : v1.22 : Update for new vsib_record
###
#######################################################################


use Astro::Coord;
use Astro::Time;
use Config::Trivial;
use Filesys::DiskFree;
use Env qw(HOME);
use Tk;
use Tk::BrowseEntry;
use Tk::Dialog;

# Debug messages
my $debug = 0;

# list of known antennas and their coordinates. Used for LST calculation
my %Antennas = (
		DSS43  => [-4460894.7273, 2682361.5296, -3674748.4238],
		DSS45  => [-4460935.4093, 2682765.7035, -3674381.2105],
		Hobart => [-3950236.7454, 2522347.5502, -4311562.5527],
		Ceduna => [-3753440.70,   3912708.30,   -3348066.90],
		Mopra  => [-4682768.6300, 2802619.0600, -3291759.9000],
		ATCA   => [-4751630.5370, 2791692.8690, -3200483.7470], #station W104
		Parkes => [-4554232.1122, 2816759.1425, -3454036.0058],
		Test   => [-4751630.5370, 2791692.8690, -3200483.7470], # Same as ATCA

);
# Antenna abbreviations. Used for naming data files
my %Antenna_abbrev = (
		DSS43  => "43",
		DSS45  => "45",
		Hobart => "Ho",
		Ceduna => "Cd",
		Mopra  => "Mp",
		ATCA   => "At",
		Parkes => "Pk",
		Test => "Tt"
);
my @template_comments = ("--- Select ---",
			 "Antenna is on point",
			 "Antenna if off source",
			 "Antenna moving off source for pointing check",
			 "Wind stow. Antenna off point.",
			 "Manchester United 3, Arsenal 0"
			 );
my $template_comment = $template_comments[0];
my @available_compress_modes = ("oooo", "ooox", "ooxx", "oxox", "xxxx");
my @data_rate_compress_factor = (0, 0.25, 0.5, 0.5, 1);
my @obs_purposes = ("Test", "Fringe check", "Science");
my @bandwidths = (1, 2, 4, 8, 16, 32, 64);
my $bandwidth = 16;
my @available_number_of_bits = (8, 16);
#############################################################################
# Standard mode definitions:
# These are the available modes. For everything except "Other"
# the number of bits and the compression modes are fixed
my @rec_modes = ("Chans 1, 2", "Chans 1, 3", "Chans 1 -> 4", "Chans 1 -> 8", 
		 "Chan 1,2 & 5, 6", "Other");
my @number_of_bits = (8, 8, 8, 16, 16, 16);
my @compress_modes = ("ooxx", "oxox", "xxxx", "xxxx", "ooxx", "xxxx");
#############################################################################

# defaults
my $rec_mode = $rec_modes[2];
my $nbits = $number_of_bits[2];
my $compress = $compress_modes[2];
my $start_mjd = now2mjd();
my $record_time_days = 0.5;
my $stop_mjd = $start_mjd + $record_time_days;
my $rate = 256;
my $scheduled_start = 1;
my $stop_recording = 0;
my $experiment = "VT00";
my $obs_purpose = $obs_purposes[0];
#maintain an array which remembers the last $n_pps_remember 1pps OK/not OK results
my $n_pps_remember = 100;
my @pps_history;
# start warning if the number of missed 1pps in the buffer gets above $n_pps_warn
my $n_pps_warn = 5;
# this variable gets set to 1 if $n_pps_warn is exceeded
my $too_many_missed_pps = 0;
# this value gets set to 1 when the user has acknowledged the problem
my $too_many_missed_pps_msg_started = 0;
# last bugbuf size and %
my $last_bigbuf_size_MB = 0;
my $last_bigbuf_free_percent = 100;

# this value gets set to 1 when a message describing a problem with
# data files not being written has been sent to the user
my $missed_file_msg_started = 0;
# this value gets set to 1 when a message describing a problem with
# the active disk getting too full has been sent to the user
my $disk_full_msg_started = 0;
# this value gets set to 1 when a message describing a problem with
# bigbuf getting used up has been sent to the user
my $bigbuf_panic_msg_started = 0;

my $errorwin_free = 1;
# signals:
my $start_pps_dialog = 0;
my $start_bigbuf_dialog = 0;
my $start_fulldisk_dialog = 0;
my $start_missing_file_dialog = 0;


# global variables
my ($ut_str, $lst_str, $main, $tv_data_file, $log_t, $comment, $ok_button, $logfile);
my ($nbits_entry, $compress_entry);
my ($nbits_label, $compress_label);
my ($start_y, $start_d, $start_h, $start_m, $start_s);
get_start_ydhms();
my ($dur_d, $dur_h, $dur_m, $dur_s) = (0,0,1,0);
my $pid; # PID of vsib_record
# GUI defaults #########################################################
my $font = '*-helvetica-medium-r-*-*-12-*';
my $boldfont = '*-helvetica-bold-r-*-*-12-*';
my $value_bg = 'grey90';
my $fg = 'black';
my $entry_bg = 'white';
my ($y_e, $d_e, $h_e, $m_e, $s_e, $start_button, $stop_button);
my $dialog;
my $tv_stats;
my $tv_rec_stat = "Recorder Idle";
my $tv_pps_n_bad = "-";
my $tv_pps_n = "-";
my $tv_last_pps = "n/a";
my $tv_last_bigbuf = "n/a";
my $tv_last_block = "n/a";
update_status_msg();
my $cycle_count = 0;
my $tv_cycle;
#cycle_heartbeat();
my ($errorwin, $err_msg, $err_frame1);
# read configuration file
my $config = Config::Trivial->new(config_file => "$HOME/.disko_config");
my $settings = $config->read;
# check that the settings make sense and die if there's something wrong
if (!check_settings($settings)) {
  die "$progname exiting due to configuration problem\n";
}

# open the log file
# 
$logfile = sprintf "Disko_%s_%s.log",$settings->{'antenna'},get_logfile_timestr();
if (!open_log($settings->{'log_dir'}, $logfile)) {
  die "$progname exiting due to problem with the log file\n";
}

# get antenna coordinates and convert to lat, long
my @ant_coord = @{$Antennas{$settings->{'antenna'}}};
my ($long_turns, $lat_turns) = Astro::Coord::r2pol(@ant_coord);
my $lat_deg = $lat_turns*360.0;
my $long_deg = $long_turns*360.0;
if ($debug) {
  print "lat = $lat_deg, long = $long_deg\n";
}

# get data directories and calculate capacity of each one
my @data_disks = split " ", $settings->{'disks'};

my $disk_handle = new Filesys::DiskFree;
my $active_disk;
my @disk_avail;
my @disk_total;
my @disk_used;
my @disk_perc_free;
my @disk_time_left;
get_disk_stats($disk_handle, \@disk_avail, \@disk_total, \@disk_used, \@disk_perc_free, \@disk_time_left);



# ready to start the GUI now
init_gui();

# start the main loop
MainLoop();

sub check_settings {
  my ($settings) = @_;
  my $key;
  my $errmsg = "";
  my $ok = 1;
  # go through each parameter and check that it makes sense

  # antenna
  my $found_antenna = 0;
  foreach $key (keys %Antennas) {
    if ($debug) {print "key = $key\n";}
    if ($key eq $settings->{'antenna'}) {
      $found_antenna = 1;
    } 
  }
  if (!$found_antenna) {
    $errmsg = $errmsg."Antenna given in config file is not known. Should be one of\nthe following:\n";
    foreach $key (keys %Antennas) {
      $errmsg = $errmsg."\t$key\n";
    }
    $errmsg = $errmsg."Note the antenna names are case sensitive.\n\n";
    $ok = 0;
  }

  # log file directory

  # data disk directories

  # max_capacity

  # recorder file size

  # recorder block size

  # vsib device

  if (!$ok) {
    print "$errmsg";
  }
  return ($ok);
}

sub open_log {
  my ($dir,$file) = @_;
  my $ok = 1;
  my $logfile = "$dir/$file";
  if (!open (LOG, "> $logfile")) {
    $ok = 0; 
    print "\nCould not open log file $logfile: $!\n";
  }
  return($ok);
}






sub init_gui {
  $ut_str = "00:00:00";
  $lst_str = "01:01:01";
  my $hostname = `hostname`;
  chomp $hostname;

  if ($debug) {print "Starting GUI...\n";}
  $main = MainWindow->new;
  $main->title("$progname v$progvers on $hostname");
  $main->iconname("$progname");
  $main->minsize(5, 5);
  $main->positionfrom('user');
  $main->protocol('WM_DELETE_WINDOW' => \&exit_program);

  ######################################################################
  ## Frames that contain various logical subsections

  my $menu_frame = $main->Frame(-relief => 'raised', 
				-borderwidth => 2,
				-background => $value_bg)
    ->grid(-row => 0, -column => 0, -columnspan =>4, -sticky => 'we');
  my $time_ant_frame = $main->Frame(-relief => 'flat', 
				-borderwidth => 0,
				-background => $value_bg)
    ->grid(-row => 10, -column => 0, -columnspan =>4, -sticky => 'ew');
  my $disks_frame = $main->Frame(-relief => 'groove', 
					-borderwidth => 2,
				 -background => $value_bg)
    ->grid(-row => 30, -column => 0, -columnspan =>4, -sticky => 'ew');
  my $recmode_frame = $main->Frame(-relief => 'groove', 
				-borderwidth => 2,
				   -background => $value_bg)
    ->grid(-row => 40, -column => 0, -columnspan =>4, -sticky => 'ew');
  my $recording_frame = $main->Frame(-relief => 'groove', 
				     -borderwidth => 2,
				     -background => $value_bg)
    ->grid(-row => 50, -column => 0, -columnspan =>4, -sticky => 'ew');
#raised, sunken, flat, ridge, solid, and groove.
  my $button_frame = $main->Frame(-relief => 'ridge', 
				  -borderwidth => 3,
				  -background => $value_bg)
    ->grid(-row => 55, -column => 0, -columnspan =>4, -sticky => 'ew');
  my $stats_frame = $main->Frame(-relief => 'groove', 
				-borderwidth => 2,
				 -background => $value_bg)
    ->grid(-row => 60, -column => 0, -columnspan =>4, -sticky => 'ew');
  my $log_frame = $main->Frame(-relief => 'groove', 
			       -borderwidth => 0,
			       -background => $value_bg)
    ->grid(-row => 70, -column => 0, -columnspan =>4, -sticky => 'ew');
  my $add_log_comment_frame = $main->Frame(-relief => 'groove', 
					-borderwidth => 0,
					   -background => $value_bg)
    ->grid(-row => 80, -column => 0, -columnspan =>4, -sticky => 'ew');
  my $status_and_messages_frame = $main->Frame(-relief => 'groove', 
					-borderwidth => 0,
					       -background => $value_bg)
    ->grid(-row => 90, -column => 0, -columnspan =>4, -sticky => 'ew');

  ######################################################################
  ## Items in the Menu frame
  my $menu = $menu_frame->Menubutton(-relief => 'flat',
				     -text => 'File', 
				     -underline => 0,
				     -background => $value_bg);
  $menu->command(-label => "Exit", 
		 -underline => 0, 
		 -command => \&exit_program,
		 -background => $value_bg);
  $menu->pack(-side => 'left');
  my $cfg_menu = $menu_frame->Menubutton(-relief => 'flat',
				     -text => 'Configure', 
				     -underline => 0,
					 -background => $value_bg);
  $cfg_menu->pack(-side => 'left');
  $cfg_menu->checkbutton(-label => "Debug messages",
			 -variable => \$debug,
			 -background => $value_bg);

  ######################################################################
  ## Items in the Time frame
  $time_ant_frame->Label(-relief => 'flat',
			 -font => $boldfont,
			 -text => "Antenna :",
			 -background => $value_bg)
    ->grid(-column => 0, -row => 0, -sticky => 'e');
  $time_ant_frame->Label(-relief => 'flat',
			 -font => $font,
			 -text => "$settings->{'antenna'}",
			 -background => $value_bg)
    ->grid(-column => 1, -row => 0, -sticky => 'w');
  my $UT_box = $time_ant_frame->Label(-relief => 'flat',
				  -font => $boldfont,
				  -textvariable => \$ut_str,
				  -width => 24,
				-background => $value_bg)
    ->grid(-column => 2, -row => 0, -padx => 5, -pady => 0);
  my $LST_box = $time_ant_frame->Label(-relief => 'flat',
				  -font => $boldfont,
				   -textvariable => \$lst_str,
				   -width => 12,
				-background => $value_bg)
    ->grid(-column => 3, -row => 0, -padx => 5, -pady => 0);

  # Update the time strings every 500 ms
  $time_ant_frame->repeat(500, \&get_times);


  ######################################################################
  ## Items in the Disks frame $disks_frame
  $active_disk = $data_disks[0];
  $disks_frame->Label(-relief => 'flat', 
		      -font => $boldfont, 
		      -text => "Disk Name", 
		      -background => $value_bg)
    ->grid(-column => 1, -row => 0, -sticky => 'ew');
  $disks_frame->Label(-relief => 'flat', 
		      -font => $boldfont, 
		      -text => "Size", 
		      -background => $value_bg)
    ->grid(-column => 2, -row => 0, -sticky => 'ew');
  $disks_frame->Label(-relief => 'flat', 
		      -font => $boldfont, 
		      -text => "Remaining", 
		      -background => $value_bg)
    ->grid(-column => 3, -row => 0, -sticky => 'ew');
  $disks_frame->Label(-relief => 'flat', 
		      -font => $boldfont, 
		      -text => " % left ", 
		      -background => $value_bg)
    ->grid(-column => 4, -row => 0, -sticky => 'ew');
  $disks_frame->Label(-relief => 'flat', 
		      -font => $boldfont, 
		      -text => " Time left (h:m:s) ", 
		      -background => $value_bg)
    ->grid(-column => 5, -row => 0, -sticky => 'ew');
  my $r;
  for ($r = 1; $r <= $#data_disks+1; $r++) {
    $disks_frame->Radiobutton(-text => "$data_disks[$r-1]", 
			      -font => $font, 
			      -variable => \$active_disk,
			      -value => $data_disks[$r-1],
			      -command => sub {logit("Data will be written to $active_disk at the next [Start]\n");},
			      -background => $value_bg)
      ->grid(-column => 1, -row => $r, -sticky => 'w');
    $disks_frame->Label(-relief => 'flat', 
			-font => $font, 
			-textvariable => \$disk_total[$r-1],
			-background => $value_bg)
      ->grid(-column => 2, -row => $r, -sticky => 'ew');
    $disks_frame->Label(-relief => 'flat', 
			-font => $font, 
			-textvariable => \$disk_avail[$r-1],
			-background => $value_bg)
      ->grid(-column => 3, -row => $r, -sticky => 'ew');
    $disks_frame->Label(-relief => 'flat', 
			-font => $font, 
			-textvariable => \$disk_perc_free[$r-1],
			-background => $value_bg)
      ->grid(-column => 4, -row => $r, -sticky => 'ew');
    $disks_frame->Label(-relief => 'flat', 
			-font => $font, 
			-textvariable => \$disk_time_left[$r-1],
			-background => $value_bg)
      ->grid(-column => 5, -row => $r, -sticky => 'ew');
  }
  ######################################################################
  ## Items in the Recorder mode frame $recmode_frame
  $recmode_frame->Label(-relief => 'flat',
			 -font => $boldfont,
			 -text => "     Experiment :",
			 -background => $value_bg)
    ->grid(-column => 0, -row => 0, -sticky => 'e');
  $recmode_frame->Entry(-width => 8, 
			 -textvariable => \$experiment, 
			 -font => $font, 
			 -background => $entry_bg)
    ->grid(-column => 1, -row => 0, -sticky => 'w');
  $recmode_frame->Label(-relief => 'flat',
			-font => $boldfont,
			-text => "Bandwidth per channel (MHz)",
			-background => $value_bg)
    ->grid(-column => 0, -row => 1, -sticky => 'e');
  $recmode_frame->BrowseEntry(
			      -choices => \@bandwidths,
			      -variable => \$bandwidth,
			      -autolimitheight => 'true',
			      -listheight => ($#bandwidths+1),
			      -autolistwidth => 'true',
			      -listwidth => 10,
			      -width => 3,
			      -background => $value_bg,
			      -browsecmd => \&set_mode_params,
			      -state => 'readonly'
			      
			      )
    ->grid(-column => 1, -row => 1, -sticky => 'w');
  $recmode_frame->Label(-relief => 'flat',
			-font => $boldfont,
			-text => "Record mode",
			-background => $value_bg)
    ->grid(-column => 0, -row => 2, -sticky => 'e');
  $recmode_frame->BrowseEntry(
			      -choices => \@rec_modes,
			      -variable => \$rec_mode,
			      -autolimitheight => 'true',
			      -listheight => ($#rec_modes+1),
			      -autolistwidth => 'true',
			      -listwidth => 35,
			      -width => 14,
			      -background => $value_bg,
			      -state => 'readonly',
			      -browsecmd => \&set_mode_params
			      
			      )
    ->grid(-column => 1, -row => 2, -sticky => 'w');
  $nbits_label = $recmode_frame->Label(-relief => 'flat',
			-font => $boldfont,
			-text => "Number of bits",
			-background => $value_bg)
    ->grid(-column => 2, -row => 2, -sticky => 'e');
  $nbits_entry = $recmode_frame->BrowseEntry(
			      -choices => \@available_number_of_bits,
			      -variable => \$nbits,
			      -autolimitheight => 'true',
			      -listheight => ($#available_number_of_bits+1),
			      -autolistwidth => 'true',
			      -listwidth => 30,
			      -width => 4,
			      -background => $value_bg,
			      -state => 'disabled',
			      -browsecmd => \&set_mode_params
			      )
    ->grid(-column => 3, -row => 2, -sticky => 'w');
  $compress_label = $recmode_frame->Label(-relief => 'flat',
			-font => $boldfont,
			-text => "Compression mode",
		-background => $value_bg)
    ->grid(-column => 2, -row => 3, -sticky => 'e');
  $compress_entry = $recmode_frame->BrowseEntry(
			      -choices => \@available_compress_modes,
			      -variable => \$compress,
			      -autolimitheight => 'true',
			      -listheight => ($#available_compress_modes+1),
			      -autolistwidth => 'true',
			      -listwidth => 30,
			      -width => 4,
			      -background => $value_bg,
			      -state => 'disabled',
			      -browsecmd => \&set_mode_params
			      )
    ->grid(-column => 3, -row => 3, -sticky => 'w');
  $recmode_frame->Label(-relief => 'flat',
			-font => $boldfont,
			-text => "Bit rate (Mbps)",
			-background => $value_bg)
    ->grid(-column => 0, -row => 3, -sticky => 'e');
  $recmode_frame->Label(-relief => 'flat',
			-font => $font,
			-textvariable => \$rate,
			-background => $value_bg)
    ->grid(-column => 1, -row => 3, -sticky => 'w');

  # Update the disk stats every 30 sec
  set_mode_params();
  $disks_frame->repeat(10000, sub {get_disk_stats($disk_handle, \@disk_avail, \@disk_total, \@disk_used, \@disk_perc_free, \@disk_time_left);});

  ######################################################################
  ## Items in the Recording control frame $recording_frame
  # Observation type: scheduled or manual start
  $recording_frame->Label(-relief => 'flat',
		       -text => "Observation type :", -font => $boldfont,
		    -background => $value_bg)
    ->grid(-column => 0, -row => 0, -sticky => 'w');
  $recording_frame->Radiobutton(-text => "Scheduled start", 
				-font => $font, 
				-variable => \$scheduled_start,
				-value => 1,
				-command => \&config_sched_widgets,
				-background => $value_bg)
    ->grid(-column => 1, -row => 0, -sticky => 'w');
  $recording_frame->Radiobutton(-text => "Manual start", 
				-font => $font, 
				-variable => \$scheduled_start,
				-value => 0,
				-command => \&config_sched_widgets,
				-background => $value_bg)
    ->grid(-column => 1, -row => 1, -sticky => 'w');

  $recording_frame->Label(-relief => 'flat',
		       -text => "    Start time :", -font => $boldfont,
		    -background => $value_bg)
    ->grid(-column => 2, -row => 0, -sticky => 'w');
  $recording_frame->Label(-relief => 'flat',
			  -text => "Year:", 
			  -font => $font,
			  -background => $value_bg)
    ->grid(-column => 3, -row => 0, -sticky => 'w');
  $y_e = $recording_frame->Entry(-width => 4, 
			  -textvariable => \$start_y, 
			  -font => $font, 
			  -background => $entry_bg)
    ->grid(-column => 4, -row => 0, -sticky => 'w');
  $recording_frame->Label(-relief => 'flat',
			  -text => "Day:", 
			  -font => $font,
			  -background => $value_bg)
    ->grid(-column => 5, -row => 0, -sticky => 'w');
  $d_e = $recording_frame->Entry(-width => 3, 
			  -textvariable => \$start_d, 
			  -font => $font, 
			  -background => $entry_bg)
    ->grid(-column => 6, -row => 0, -sticky => 'w');
  $recording_frame->Label(-relief => 'flat',
			  -text => "h:", 
			  -font => $font,
			  -background => $value_bg)
    ->grid(-column => 7, -row => 0, -sticky => 'w');
  $h_e = $recording_frame->Entry(-width => 2, 
			  -textvariable => \$start_h, 
			  -font => $font, 
			  -background => $entry_bg)
    ->grid(-column => 8, -row => 0, -sticky => 'w');
  $recording_frame->Label(-relief => 'flat',
			  -text => "m:", 
			  -font => $font,
			  -background => $value_bg)
    ->grid(-column => 9, -row => 0, -sticky => 'w');
  $m_e = $recording_frame->Entry(-width => 2, 
			  -textvariable => \$start_m, 
			  -font => $font, 
			  -background => $entry_bg)
    ->grid(-column => 10, -row => 0, -sticky => 'w');
  $recording_frame->Label(-relief => 'flat',
			  -text => "s:", 
			  -font => $font,
			  -background => $value_bg)
    ->grid(-column => 11, -row => 0, -sticky => 'w');
  $s_e = $recording_frame->Entry(-width => 2, 
			  -textvariable => \$start_s, 
			  -font => $font, 
			  -background => $entry_bg)
    ->grid(-column => 12, -row => 0, -sticky => 'w');

  $recording_frame->Button(-text => 'Now',
			   -font => $font,
			   -width => 3,
			   -command => \&get_start_ydhms,
			   -background => $value_bg)
    ->grid(-column => 13, -row => 0, -sticky => 'w');
#
  $recording_frame->Label(-relief => 'flat',
		       -text => "    Duration :", -font => $boldfont,
		    -background => $value_bg)
    ->grid(-column => 2, -row => 1, -columnspan => 3, -sticky => 'e');
  $recording_frame->Label(-relief => 'flat',
			  -text => "Days:", 
			  -font => $font,
			  -background => $value_bg)
    ->grid(-column => 5, -row => 1, -sticky => 'w');
  $recording_frame->Entry(-width => 3, 
			  -textvariable => \$dur_d, 
			  -font => $font, 
			  -background => $entry_bg)
    ->grid(-column => 6, -row => 1, -sticky => 'w');
  $recording_frame->Label(-relief => 'flat',
			  -text => "h:", 
			  -font => $font,
			  -background => $value_bg)
    ->grid(-column => 7, -row => 1, -sticky => 'w');
  $recording_frame->Entry(-width => 2, 
			  -textvariable => \$dur_h, 
			  -font => $font, 
			  -background => $entry_bg)
    ->grid(-column => 8, -row => 1, -sticky => 'w');
  $recording_frame->Label(-relief => 'flat',
			  -text => "m:", 
			  -font => $font,
			  -background => $value_bg)
    ->grid(-column => 9, -row => 1, -sticky => 'w');
  $recording_frame->Entry(-width => 2, 
			  -textvariable => \$dur_m, 
			  -font => $font, 
			  -background => $entry_bg)
    ->grid(-column => 10, -row => 1, -sticky => 'w');
  $recording_frame->Label(-relief => 'flat',
			  -text => "s:", 
			  -font => $font,
			  -background => $value_bg)
    ->grid(-column => 11, -row => 1, -sticky => 'w');
  $recording_frame->Entry(-width => 2, 
			  -textvariable => \$dur_s, 
			  -font => $font, 
			  -background => $entry_bg)
    ->grid(-column => 12, -row => 1, -sticky => 'w');
 
  ######################################################################
  ## Items in the start/stop frame $button_frame
  $start_button = $button_frame->Button(-text => 'Start',
			   -font => $boldfont,
			   -width => 10,
			   -command => \&start_recording,
			   -background => 'light green')
    ->grid(-column => 0, -row => 0, -columnspan => 1, -sticky => 'ew');
  $stop_button = $button_frame->Button(-text => 'Stop',
			   -font => $boldfont,
			   -width => 10,
			   -command => \&stop_recording,
			   -background => 'pink')
    ->grid(-column => 1, -row => 0, -columnspan => 1,-sticky => 'ew');
  $start_button->configure(-state => 'normal');
  $stop_button->configure(-state => 'disable');

  ######################################################################
  ## Items in the Recorder statistics frame $stats_frame
  $stats_frame->Label(-relief => 'flat',
		      -text => "Recorder monitoring :", -font => $boldfont,
		      -background => $value_bg)
    ->grid(-column => 0, -row => 0, -sticky => 'w');
#  $stats_frame->Label(-textvariable => \$tv_cycle,
#		      -width => 1,
#		      -font => $font, 
#		      -relief => 'flat',
#		      -background => $value_bg)
#    ->grid(-column => 1, -row => 0, -sticky => 'ew');


  $stats_frame->Label(-relief => 'flat',
		      -text => "vsib_record status :", -font => $boldfont,
		      -background => $value_bg)
    ->grid(-column => 0, -row => 1, -sticky => 'w');
  $stats_frame->Label(-textvariable => \$tv_rec_stat,
		      -width => 10,
		      -font => $font, 
		      -relief => 'flat',
		      -background => $value_bg)
    ->grid(-column => 1, -row => 1, -sticky => 'ew');

  $stats_frame->Label(-relief => 'flat',
		      -text => "Last 1PPS :", -font => $boldfont,
		      -background => $value_bg)
    ->grid(-column => 0, -row => 2, -sticky => 'w');
  $stats_frame->Label(-textvariable => \$tv_last_pps,
		      -width => 8,
		      -font => $font, 
		      -relief => 'flat',
		      -background => $value_bg)
    ->grid(-column => 1, -row => 2, -sticky => 'ew');

  $stats_frame->Label(-relief => 'flat',
		      -text => "Missed 1PPS :", -font => $boldfont,
		      -background => $value_bg)
    ->grid(-column => 2, -row => 2, -sticky => 'w');
  $stats_frame->Label(-textvariable => \$tv_stats,
		      -width => 18,
		      -font => $font, 
		      -relief => 'flat',
		      -background => $value_bg)
    ->grid(-column => 3, -row => 2, -sticky => 'ew');

  $stats_frame->Label(-relief => 'flat',
		      -text => "Last Bigbuf :", -font => $boldfont,
		      -background => $value_bg)
    ->grid(-column => 0, -row => 3, -sticky => 'w');
  $stats_frame->Label(-textvariable => \$tv_last_bigbuf,
		      -width => 3,
		      -font => $font, 
		      -relief => 'flat',
		      -background => $value_bg)
    ->grid(-column => 1, -row => 3, -sticky => 'ew');
  $stats_frame->Label(-relief => 'flat',
		      -text => "Last block :", -font => $boldfont,
		      -background => $value_bg)
    ->grid(-column => 2, -row => 3, -sticky => 'w');
  $stats_frame->Label(-textvariable => \$tv_last_block,
		      -width => 18,
		      -font => $font, 
		      -relief => 'flat',
		      -background => $value_bg)
    ->grid(-column => 3, -row => 3, -sticky => 'ew');

  ######################################################################
  ## Items in the Log frame $log_frame
  $log_frame->Label(-relief => 'flat',
		       -text => "Log :", -font => $boldfont,
		    -background => $value_bg)
    ->pack(-anchor => 'w');
#  $log_t = $log_frame->Scrolled(qw/Text -relief sunken 
#				-borderwidth 2 -setgrid true -width 80
#				-height 10 -background grey90 -scrollbars e/)
#    ->pack(-expand => 1, -fill =>  'both');
  $log_t = $log_frame->Scrolled('Text',
				-relief => 'sunken',
				-borderwidth => 2,
				-setgrid => 'true',
				-width => 80,
				-height => 10,
				-background => $value_bg,
				-scrollbars => 'e')
    ->pack(-expand => 1, -fill =>  'both');
  $log_t->mark(qw/set insert 0.0/);
  $log_t->tagConfigure('r', 'foreground', 'red');
  $log_t->tagConfigure('g', 'foreground', 'green');
  $log_t->tagConfigure('b', 'foreground', 'blue');
  $log_t->tagConfigure('y', 'foreground', 'blue', 'background', 'yellow');
  my $time =get_log_timestr();
  $log_t->insert('0.0',"$time $progname version $progvers.\n",'y');
  print LOG "$time $progname version $progvers.\n";
  $log_t->insert('end',"$time Log file $logfile opened.\n");
  print LOG "$time Log file $logfile opened.\n";

  ######################################################################
  ## Items in the Add Log Comment frame

  # Comment entry:
  $add_log_comment_frame->Label(-relief => 'flat',
		       -text => "Comment :", -font => $boldfont,
				-background => $value_bg)
    ->grid(-row =>0, -column => 0, -columnspan => 1, -sticky => 'e');
  my $comment_entry = $add_log_comment_frame->Entry(-width => 70, 
						    -textvariable => \$comment, 
						    -font => $font, 
						    -background => $entry_bg)
      ->grid(-row =>0, -column => 1, -columnspan => 39, -sticky => 'ew');
  $ok_button = $add_log_comment_frame->Button(-text    => "OK", -font => $boldfont,
					      -width => 4, -justify => 'left',
					-command => \&enter_comment,
					      -background => $value_bg)
    ->grid(-row => 0, -column => 41, -sticky => 'e');
  $comment_entry->bind("<Return>",\&enter_comment);

  $add_log_comment_frame->Label(-relief => 'flat',
			-font => $boldfont,
			-text => "Template comments",
			-background => $value_bg)
    ->grid(-column => 0, -row => 2, -sticky => 'e');
  $add_log_comment_frame->BrowseEntry(
				      -choices => \@template_comments,
				      -variable => \$template_comment,
				      -autolimitheight => 'true',
				      -listheight => ($#template_comments+1),
#				      -autolistwidth => 'true',
				      -listwidth => 400,
				      -width => 70,
				      -background => $value_bg,
				      -browsecmd => sub {$comment = $template_comment; 
							 $template_comment = $template_comments[0];
						       }
			      )
    ->grid(-column => 1, -row => 2, -sticky => 'w');


  $errorwin = MainWindow->new();
  $errorwin->title("Error");
  $errorwin->minsize(50, 10);
  $errorwin->geometry('+500+500');
  $errorwin->withdraw;
  $err_msg = $errorwin->Label (
			       -background => 'red',
			       -borderwidth => '5',
			       -foreground => 'yellow',
			       -padx => '4',
			       -text => "all OK",
			      )->pack;
  $err_frame1 = $errorwin->Frame (
				  -background => 'red',
				  -borderwidth => '0',
				 )->pack(-fill => 'x');
  $err_frame1->Button (
		       -activebackground => 'magenta',
		       -activeforeground => 'yellow',
		       -background => 'red',
		       -borderwidth => '5',
		       -foreground => 'yellow',
		       -relief => 'groove',
		       -command => sub {$errorwin->withdraw;$errorwin_free=1},
		       -text => 'OK',
		       )->pack();



  $main->repeat(510, \&start_dialogs);


}

sub get_times {
  my $mjd = now2mjd();
  my $dayno;
  my ($day, $month, $year, $ut) = mjd2cal($mjd);
  ($dayno, $year, $ut) = mjd2dayno($mjd);
  my $str = turn2str($ut, 'H', 0);
  $ut_str = sprintf ("%d day %03d %s UT",$year,$dayno,$str);
  my $lst = mjd2lst($mjd, deg2turn($long_turns));
  $str = turn2str($lst, 'H', 0);
  $lst_str = sprintf ("%s LST",turn2str($lst, 'H', 0));
}



sub exit_program {
  # close_log
  close LOG;
  exit;
}

sub enter_comment {
 my $time =get_log_timestr();
 if (defined($comment)) {
     $comment = $time." ".$comment;
     $log_t->insert('end',"$comment\n");
     print LOG "$comment\n";
     $comment = "";
     $log_t->yview(moveto => 1);
     $log_t->update;
 }
}


sub get_log_timestr {
  my $mjd = now2mjd();
  my $dayno;
  my ($day, $month, $year, $ut) = mjd2cal($mjd);
  ($dayno, $year, $ut) = mjd2dayno($mjd);
  my $str = turn2str($ut, 'H', 0);
  while (length($str) < 8) {
    $str = "0".$str;
  }
  my $ut_str = sprintf ("%d.%03d.%s",$year,$dayno,$str);
  return ($ut_str);
}

sub get_logfile_timestr {
  my $mjd = now2mjd();
  my $dayno;
  my ($day, $month, $year, $ut) = mjd2cal($mjd);
  ($dayno, $year, $ut) = mjd2dayno($mjd);
  my $str = turn2str($ut, 'H', 0);
  while (length($str) < 8) {
    $str = "0".$str;
  }
  $str =~ s/\://g;
  $ut_str = sprintf ("%d_%03d_%s",$year,$dayno,$str);
  return ($ut_str);
}

sub get_disk_stats {
  my ($handle, $avail_ref, $total_ref, $used_ref, $perc_ref, $time_avail_ref) = @_;
  $handle->df();
  my ($disk, $i);
  for ($i = 0; $i <= $#data_disks; $i++) {
    $handle->device($data_disks[$i]);
    $avail_ref->[$i] = sprintf "%dG",$handle->avail($data_disks[$i])/ 2**30;
    $total_ref->[$i] = sprintf "%dG",$handle->total($data_disks[$i]) / 2**30;
    $used_ref->[$i]  = sprintf "%dG",$handle->used($data_disks[$i]) / 2**30;
    $perc_ref->[$i] = sprintf "%5.1f",100.0*(1.0-($handle->used($data_disks[$i])/$handle->total($data_disks[$i])));
  }
  update_time_left($handle, $avail_ref, $time_avail_ref);
}

sub set_mode_params {
  my $i;
  my $cfact;

  if ($bandwidth == 64) {
      $rec_mode = "Other";
      $nbits = 16;
      $compress = "xxxx";
  }

  if ($rec_mode eq "Other") {
    # enable compress and bit settings
    $nbits_entry->configure(-state => 'readonly' );
    $compress_entry->configure(-state => 'readonly');
    $nbits_label->configure(-foreground => 'black');
    $compress_label->configure(-foreground => 'black');
  } else {
    for ($i = 0; $i <= $#rec_modes; $i++) {
      if ($rec_mode eq $rec_modes[$i]) {
	$nbits = $number_of_bits[$i];
	$compress = $compress_modes[$i];
      }
    }
    # disable compress and bit settings
    $nbits_entry->configure(-state => 'disable', );
    $compress_entry->configure(-state => 'disable');
    $nbits_label->configure(-foreground => 'grey50');
    $compress_label->configure(-foreground => 'grey50');
  }
  for ($i = 0; $i <= $#available_compress_modes; $i++) {
    if ($compress eq $available_compress_modes[$i]) {
      $cfact = $data_rate_compress_factor[$i];
    }
  }
  $rate = $bandwidth*$nbits*$cfact*2.0;
  $rate /= 4 if ($bandwidth==64);

  update_time_left($disk_handle, \@disk_avail, \@disk_time_left);
}

sub logit {
  my ($text) = @_;
  $text = &get_log_timestr." ".$text;
  print LOG $text;
  logmsg($text);
  if ($debug) {print "$text";}
}
sub logit_red {
  my ($text) = @_;
  $text = &get_log_timestr." ".$text;
  print LOG $text;
  logmsg_red($text);
  if ($debug) {print "$text";}
}

sub logit_blue {
  my ($text) = @_;
  $text = &get_log_timestr." ".$text;
  print LOG $text;
  logmsg_blue($text);
  if ($debug) {print "$text";}
}

sub logit_green {
  my ($text) = @_;
  $text = &get_log_timestr." ".$text;
  print LOG $text;
  logmsg_green($text);
  if ($debug) {print "$text";}
}

sub logmsg {
  $log_t->insert('end',@_);
  $log_t->yview(moveto => 1);
  $log_t->update;
}

sub logmsg_red {
  $log_t->insert('end',@_,'r');
  $log_t->yview(moveto => 1);
  $log_t->update;
}

sub logmsg_green {
  $log_t->insert('end',@_,'g');
  $log_t->yview(moveto => 1);
  $log_t->update;
}

sub logmsg_blue {
  $log_t->insert('end',@_,'b');
  $log_t->yview(moveto => 1);
  $log_t->update;
}

sub update_time_left {
  # given a data rate, calculate how much time is left on each disk
  my ($handle, $avail_ref, $time_avail_ref) = @_;
  my $i;
  for ($i = 0; $i <= $#data_disks; $i++) {
    $handle->device($data_disks[$i]);
    $time_avail_ref->[$i] = sprintf "%s",turn2str(($handle->avail($data_disks[$i])*8)/ $rate/ 1.0e+6/86400, 'H', 0);
  }

}
sub config_sched_widgets {
  if ($scheduled_start) {
    #enable
    $y_e->configure(-state => 'normal');
    $d_e->configure(-state => 'normal');
    $h_e->configure(-state => 'normal');
    $m_e->configure(-state => 'normal');
    $s_e->configure(-state => 'normal');
  } else {
    #disable
    $y_e->configure(-state => 'disable');
    $d_e->configure(-state => 'disable');
    $h_e->configure(-state => 'disable');
    $m_e->configure(-state => 'disable');
    $s_e->configure(-state => 'disable');
  }
}


sub get_start_ydhms {
  my $mjd = now2mjd();
  my $dayno;
  my ($day, $month, $year, $ut) = mjd2cal($mjd);
  ($dayno, $year, $ut) = mjd2dayno($mjd);
  my $str = turn2str($ut, 'H', 0);
  while (length($str) < 8) {
    $str = "0".$str;
  }
  $start_y = $year;
  $start_d = $dayno;
  ($start_h,$start_m,$start_s) = split ":", $str;
}

sub start_recording {
  my $mode;
  if ($nbits == 16) {
    $mode = 2;
  } else {
    $mode = 3;
  }
  my $command = "vsib_record -x -m $mode -w $bandwidth -c $compress -f $settings->{'file_size'} -b $settings->{'block_size'} -e $settings->{'vsib_device'}";
  if ($scheduled_start) {
    my ($day, $month) = dayno2cal($start_d, $start_y);
    my $ut = ($start_h+ ($start_m/60.0) + ($start_s/3600.0))/24.0;
    $command = sprintf "$command -s %4d-%02d-%02dT%02d:%02d:%02d",$start_y,$month,$day,$start_h,$start_m,$start_s;
  }

  # In case values are not filled in
  $dur_d = 0 if (!defined $dur_d || $dur_d =~ /^\s*$/);
  $dur_h = 0 if (!defined $dur_h || $dur_h =~ /^\s*$/);
  $dur_m = 0 if (!defined $dur_m || $dur_m =~ /^\s*$/);
  $dur_s = 0 if (!defined $dur_s || $dur_s =~ /^\s*$/);

  # duration. convert to sec  
  $command = sprintf "$command -t %ds",((((($dur_d*24.0) + $dur_h)*60.0) + $dur_m)*60.0) + $dur_s;
  # file name. Use the convention:
  # (data_dir)/(experiment name)/(experiment)-(antenna)-(suffix)
  $experiment = lc($experiment);
  my $subdir;

  # VT02 and vc* experiments are fringe checks and go in a separate
  # sub-dir called FringeCheck.  All other experiments go in a
  # directory with their name.
  if ($experiment =~ /^vt02/i || $experiment =~ /^vc/i) {
    $subdir = "$active_disk/FringeCheck";
  } else {
    $subdir = "$active_disk/$experiment";
  }
  if (-e $subdir) {
    logit "Data will be written to $subdir\n";
  } else {
    if (mkdir($subdir)) {
      logit "Created the directory $subdir. Data will be written here.\n";
    } else {
      logit "Could not create the directory $subdir. Writing to $active_disk instead\n";
      $subdir = "$active_disk";
    }
  }

  my $file_prefix = sprintf "%s/%s-%s",$subdir,$experiment,$Antenna_abbrev{$settings->{'antenna'}};
  $command = $command." -o $file_prefix";


  logit("Recording started with command:\n");
  logit(" $command\n");
  $stop_button->configure(-state => 'normal');
  $start_button->configure(-state => 'disable');
  $tv_rec_stat = "Recording";
  $pid = open(VSIB_RECORD, "$command |") || die "Could not exectue $command: $!\n";
  my ($rin) = ('');
  vec($rin, fileno(VSIB_RECORD), 1) = 1;
  my ($buf, $nread, $rout,$timeleft,$nfound);
  # initialise counters and timers used to measure health of the system
  @pps_history = ();
  # time since last 1PPS OKAY
  my $time_of_last_pps_sec = now2mjd()*24*3600;
  my $time_since_last_pps_sec = 0;
  my $pps_currently_ok = 1;
  my $number_of_bad_pps = 0;
  my $time_of_last_file_opened_sec = now2mjd()*24*3600;
  my $time_since_last_file_opened_sec = 0;
  my $name_of_last_file = "";
  # last block number
  my $last_block = 0;
  # dummy variable;
  my $d;

  # handle the first file opened and first good PPS.
  my $start_monitoring_pps = 0;
  my $start_monitoring_file = 0;
  my $line;
  while (!$stop_recording) {
    ($nfound,$timeleft) = select($rout=$rin,undef,undef,0.1);
#    cycle_heartbeat();
    if ($nfound > 0) {
      if (vec($rout, fileno(VSIB_RECORD), 1)) {
	if (! defined ($line=<VSIB_RECORD>)) {
	  # Process finished
	  logit_red "vsib_record process has finished.\n";
	  $stop_recording = 1;
	} else {
	  if ($debug) {
	    logit($line);
	  } else {
	    print LOG $line;
	  }
	  
	  # Process a line of output from vsib_record
	  if ($line =~ /at block/) {
	    # expecting something like this:
	    # at block = 20000, opened file 'test_255_054258.lba'
	    ($d,$d,$d,$last_block,$d,$d,$name_of_last_file) = split " ",$line;
	    $last_block =~ s/,//g;
	    $tv_last_block = $last_block;
	    $name_of_last_file =~ s/\'//g;
	    $start_monitoring_file = 1;
	    $time_of_last_file_opened_sec = now2mjd()*24*3600;
	    if ($debug) {
	      logit_blue "Last block = $last_block, last file = $name_of_last_file\n";
	    }
	  } elsif ($line =~ /PPS OKAY/) {
	    $pps_currently_ok = 1;
	    $tv_last_pps = "OK";
	    $time_of_last_pps_sec = now2mjd()*24*3600;
	    $start_monitoring_pps = 1;
	    # update the pps history array. Send 0 = OK, 1 = missed PPs
	    update_pps_history(0);
          } elsif ($line =~ /MBytes left/) {
	    # expecting something like this:
	    # MBytes left in the BIGBUF 487 (100%): 1PPS OKAY
	    ($d,$d,$d,$d,$d,$last_bigbuf_size_MB,$last_bigbuf_free_percent) = split " ",$line;
	    $last_bigbuf_free_percent =~ s/\(//;
	    $last_bigbuf_free_percent =~ s/\)//;
	    $last_bigbuf_free_percent =~ s/%//;
	    $last_bigbuf_free_percent =~ s/://;
	    $tv_last_bigbuf = "$last_bigbuf_free_percent%";
	    if ($debug) {
	      logit_blue "PPS OK = $pps_currently_ok, last_bigbuf_free_percent = $last_bigbuf_free_percent\n";
	    }
	    # check BIGBUF size
	    if ($last_bigbuf_free_percent >= $settings->{'bigbuf_warn_limit'}) {
	      # bigbuf OK. Set default parameters
	      if ($bigbuf_panic_msg_started) {
		logit "Bigbuf levels OK: $last_bigbuf_free_percent%\n";
	      }
	      $bigbuf_panic_msg_started = 0;
	      $start_bigbuf_dialog = 0;
	    } elsif ($last_bigbuf_free_percent < $settings->{'bigbuf_panic_limit'}) {
	      # bring up a dialog and beep
	      logit_red "Bigbuf level is very low: $last_bigbuf_free_percent%\n";
	      if (!$bigbuf_panic_msg_started) {
		$bigbuf_panic_msg_started = 1;
		$start_bigbuf_dialog = 1;
	      }
	    } else {
	      # bigbuf is between bigbuf_warn_limit and bigbuf_panic_limit
	      logit_red "Bigbuf level is low: $last_bigbuf_free_percent%\n";
	      $bigbuf_panic_msg_started = 0;
	      $main->bell();
	    }
	    
	    
	  } elsif ($line =~ /1PPS transition ABSENT/) {
	    # a bad 1pps seen
	    logit_red "1PPS Missed!\n";
	    $tv_last_pps = "Missed";
	    $main->bell();
	    $number_of_bad_pps++;
	    $pps_currently_ok = 0;
	    $start_monitoring_pps = 1;
	    # update the pps history array. Send 0 = OK, 1 = missed PPs
	    update_pps_history(1);
	  } 
	  $time_since_last_pps_sec = now2mjd()*24*3600 - $time_of_last_pps_sec;
	  $time_since_last_file_opened_sec = now2mjd()*24*3600 - $time_of_last_file_opened_sec;
	  
	  # now check if any of the time limits have expired
	  if ($start_monitoring_pps) {
	    if ($debug) {
	      logit_blue("time_since_last_pps_sec = $time_since_last_pps_sec\n");
	    }
	    if ($too_many_missed_pps && !$too_many_missed_pps_msg_started) {
	      # bring up a dialog and beep
	      print "\n\n\nstart pps dialog\n\n\n";
	      if (!$start_pps_dialog) {$start_pps_dialog = 1;}
	    }
	  }
	  if ($start_monitoring_file) {
	    if ($debug) {
	      logit_blue("time_since_file_opened_sec = $time_since_last_file_opened_sec\n");
	    }
	    my $max_time_between_files = $settings->{'file_size'};
	    $max_time_between_files =~ s/s//;
	    # add a couple of seconds to this, just to allow for latency
	    $max_time_between_files += 2;
	    if (($time_since_last_file_opened_sec > $max_time_between_files) &&
		!$missed_file_msg_started){  
		if ($debug) {
		    print "\tmissed file TRIGGER $errorwin_free, $time_since_last_file_opened_sec > $max_time_between_files, $missed_file_msg_started\n";
		}
	      # bring up a dialog and beep
	      $missed_file_msg_started = 1;
	      $start_missing_file_dialog = 1;
	    } elsif ($time_since_last_file_opened_sec <= $max_time_between_files) {
		if ($debug) {
		    print "\t missed file no trigger $errorwin_free, $time_since_last_file_opened_sec > $max_time_between_files, $missed_file_msg_started\n";
		}
	      $missed_file_msg_started = 0;
	    }
	  }
	  # what's the capacity of the active disk?
	  
	  $disk_handle->device($active_disk);
	  my $perc_full = 100.0* $disk_handle->used($active_disk)/
	    $disk_handle->total($active_disk);
	  if ($perc_full > $settings->{'max_capacity'}) {
	    # volume of active disk is greater than the limit
	    # bring up a dialog and beep
	    if (!$disk_full_msg_started) {
	      $start_fulldisk_dialog = 1;
	      $disk_full_msg_started = 1;
	    }
	    $d = sprintf "Disk space getting low: %5.1f percent.\n",$perc_full;
	    logit_red $d;
	  } else {
	    if ($disk_full_msg_started) {
	      $d = sprintf "Disk space now OK: %5.1f percent.\n",$perc_full;
	      logit_red $d;
	    }
	    $disk_full_msg_started = 0;
	  }
	  
	  
	}
	
      } else {
	print "novec\n";
      }
    } else {
      # waiting for line from stdout
      if ($debug) {print ".";}
    }
    $main->update;
  }
  # if we've got this far, the signal to stop recording has been receieved
  close(VSIB_RECORD);
  $stop_recording = 0;
  logit("Recording stopped\n");
  $tv_rec_stat = "Idle";
  $start_button->configure(-state => 'normal');
  $stop_button->configure(-state => 'disable');


}

sub stop_recording {
  # this subroutine is called when the Stop button is pressed. 
  # all it needs to do is change the value of $stop_recording to 1. The
  # start_recording subroutine should be in a while (!$stop_recording) loop
  # so when it sees the change it will finish the loop, tell vsib_record
  # to stop and return.
  $stop_recording = 1;
  # send Ctrl-C to VSIB_RECORD
  kill 2, $pid;
  logit("Stop request received\n");
}

sub update_pps_history {
  my ($flag) = @_;
  push @pps_history, $flag;
  if ($#pps_history > ($n_pps_remember-1)) {
    # the array has reached maximum. Forget the oldest record
    shift @pps_history;
  }
  # count the number of bad 1pps
  my ($i,$j) = (0,0);
  foreach $j (@pps_history) {
    $i += $j;
  }
  $tv_pps_n_bad = $i;
  $tv_pps_n = $#pps_history+1;
  update_status_msg();
  if ($i > $n_pps_warn) {
    # set the flag to warn the user if the number of missed pps's gets too high.
    $too_many_missed_pps = 1;
#    print "\t\tPPS BAD $i\n"
  } else {
    # otherwise set things to all-OK.
    $too_many_missed_pps = 0;
    $too_many_missed_pps_msg_started = 0;
#    print "\t\tPPS OK $i\n"
  }

}

sub start_dialogs {
  # depending on signals, will start appropriate dialogs
  if ($errorwin_free) {
    if ($start_pps_dialog) {
      $too_many_missed_pps_msg_started = 1;
      $start_pps_dialog = 0;
      tk_error("More than $n_pps_warn of the last $n_pps_remember 1PPS checks have been missed.\nPlease investigate the problem\n");
      $errorwin_free = 0;
    } elsif ($start_bigbuf_dialog) {
      $bigbuf_panic_msg_started = 1;
      $start_bigbuf_dialog = 0;
      tk_error("The capacity of BIGBUF is now $last_bigbuf_free_percent% which is very low.\n Vsib_record may die soon and requre restarting.\n Consider stopping recording and investigate the problem\n");
      $errorwin_free = 0;
    } elsif ($start_fulldisk_dialog) {
      $disk_full_msg_started = 1;
      $start_fulldisk_dialog = 0;
      my $d = sprintf "Disk capacity has exceeded %d percent.\nConsider deleting old files or stopping recording and changing disks.\n",$settings->{'max_capacity'};
      tk_error($d);
      $errorwin_free = 0;
    } elsif ($start_missing_file_dialog) {
      $missed_file_msg_started = 1;
      $start_missing_file_dialog = 0;
      tk_error("Vsib_record has missed opening a data file.\nThis is probably a serious problem, please investigate.\n");
      $errorwin_free = 0;
    }
  }
}




sub tk_error {
  $err_msg->configure(-text => @_);
  $errorwin->deiconify;
  $err_msg->bell;
  $err_msg->bell;
  $err_msg->bell;
  $err_msg->bell;
  $err_msg->bell;
}


sub update_status_msg {
    $tv_stats = "$tv_pps_n_bad from last $tv_pps_n";

}

sub cycle_heartbeat {
    my @cycle = (".","o","O","o");
    $cycle_count++;
    if ($cycle_count > $#cycle) {
	$cycle_count = 0;
    }
    $tv_cycle = $cycle[$cycle_count];
}
