#!/usr/bin/perl -w

use strict;
use Carp;

use IO::Socket;
use Getopt::Long;

use constant M5DATA  => 2630;
use constant M5DRIVE => 2620;
use constant RECORDER_SERVER => 50080;

sub send_data($);
sub send_cmd($);

my $ant = defined($ENV{RTFC_ANTID}) ? $ENV{RTFC_ANTID}: 'Test';
my $chans = 'xxxx';
my $bandwidth = 16;
my $vsib_mode = 2;


my $m5data = M5DATA;
my $mtu = 1500;
my $udp = 0;
my $onebit = 0;
my $invert = 0;
my $debug = 0;


my $recordername = 'localhost';

GetOptions('recorder=s'=>\$recordername, 'onebit'=>\$onebit,
	   'invert'=>$invert, 'debug'=>\$debug);

my $listensock = new IO::Socket::INET(LocalPort => M5DRIVE,
				      Listen => 1,
				      ReuseAddr => 1,
				     )
  || die "Couldn't create socket: $!\n";


# Connect to the recorder_server and setup the recording mode

my @setup_data = (qw(round_start=off record_time=24h
		     mark5b=on filesize=2s), 
		  "filename_prefix=$ant",
		  "vsib_mode=$vsib_mode",
		  "bandwidth=$bandwidth",
		  "compression=$chans");

foreach (@setup_data) {
  my $status = send_data($_);
  die("Failed to send $_\n") if ! defined $status;
}

my $status;
if ($invert) {
  $status = send_data("invert=on");
} else {
  $status = send_data("invert=off");
}
die "Failed to send invert option"  if ! defined $status;
if ($onebit) {
  $status = send_data("onebit=on");
} else {
  $status = send_data("onebit=off");
}
die "Failed to send onebit option"  if ! defined $status;

my $requested_rate = undef;
my $protocol = undef;
my $winsize = -1;
my $ipd = 0;

my $recording = 0;
while (1) {

  print "WAiting for connection\n";
  my $sock = $listensock->accept;
  die "Failed to accept connection: $!\n" if (!$sock);

  while (<$sock>) {
    print "RECV: $_\n";
    if (/play_rate=data:(\S+)(\s*;)?/) {
      print "Requested rate of $1 Mbps\n";
      $requested_rate = $1;
      print $sock "!play_rate = 0 ;\n";

    } elsif (/play=off(\s*;)?/) {
      print "Requested to stop playing. IMPLEMENT ME!\n";
      print $sock "!play = 1 ;\n";

    } elsif (/net_protocol=(\S+):(\d+):\d+(:\d+)?(\s*;)?/) {
      $protocol = $1;
      $winsize = $2/1024;
      if ($protocol=~/tcp/i) {
	print "TCP window size set to $winsize\n";
	$udp = 0;
      } elsif ($protocol=~/udp/i) {
	$udp = 1;
	print "Protocol set to UDP\n";
      } else {
	print $sock "!net_protocol = 1 ;\n";
	die "$protocol requested - not supported\n"
      }
      print $sock "!net_protocol = 0 ;\n";

    } elsif (/mtu=(\d+)(\s*;)?/) {
      $mtu = $1;
      print "MTU set to $mtu\n";
      print $sock "!mtu = 0 ;\n";

    } elsif (/net_port=(\d+)(\s*;)?/) {
      $m5data = $1;
      print "Net_Port  $m5data\n";
      print $sock "!net_port = 0 ;\n";

    } elsif (/mtu\?/) {
      $mtu = $1;
      print "MTU state request\n";
      print $sock "!mtu? = 0 : $mtu bytes;\n";

    } elsif (/ipd=(\d+)(\s*;)?/) {
      $ipd = $1;
      print "IPD set to $ipd\n";
      print $sock "!ipd = 0 : $ipd usec;\n";

    } elsif (/in2net=connect:(\S+)(\s*;)?/) {
      print "Requested to connect to $1\n";
      
      $status = send_data("add_host=evlbi_$ant,$1,$m5data,$winsize,1");
      exit(1) if !defined $status;

      if ($udp) {
	$status = send_data("modify_host=evlbi_$ant,$mtu,$ipd");
	exit(1) if ! defined $status;
      }

      $status = send_data("diskselection=off");
      die "Failed to turn off auto disk selection\n" 
	if (!defined $status);

      print $sock "!in2net = 0 ;\n";

    } elsif (/in2net=on(\s*;)?/) {
      print "Requested to start sending data\n";

      $status = send_data("remrecorder=evlbi_$ant");
      die "Failed to set remote recorder to evlbi_$ant\n" 
	if (!defined $status);

      $status = send_cmd("record-start");
      if (!defined $status) {
	print $sock "!in2net = 1 ;\n";
      } else {
	print $sock "!in2net = 0 ;\n";
	$recording = 1;
      }

    } elsif (/in2net=disconnect(\s*;)?/) {
      print "Requested to stop sending data\n";
      my $status = send_cmd("record-stop");
      if (!defined $status){
        print $sock "!in2net = 1 ;\n";
      } else {
        print $sock "!in2net = 0 ;\n";
	$recording = 0;
      }

    } elsif (/in2net=off(\s*;)?/) {
      print "Requested to stop sending data\n";
      my $status = send_cmd("record-stop");
      if (!defined  $status) {
	print $sock "!in2net = 1 ;\n";
      } else {
	print $sock "!in2net = 0 ;\n";
	$recording = 0;
      }

    } elsif (/mode=(\S+):(\S+)(:\d+)?(\s*;)?/) {
      print "Mode requested $1:$2\n";
      print $sock "!mode = 0 ;\n";
     
    } elsif (/trackmask=(\S+)(\s*;)?/) {
      print "Trackmask requested $1\n";
      printf $sock "!trackmask = 0 : 0x%016x : 0 ;\n", 0;

    } elsif (/trackmask\?/) {
      print "trackmask status requested\n";
      printf $sock "!trackmask = 0 : 0x%016x : 0 ;\n", 0;

    } elsif (/status\?/) {
      print "Status requested\n";
      if ($recording) {
	print $sock "!status? 0 : 0x00010001 ;\n";
      } else {
	print $sock "!status? 0 : 0x00000001 ;\n";
      }

    } elsif (/play\?/) {
      print "Play state requested\n";
      print $sock "!play? 0 : off ;\n";

    } else {
      print "Unexpected command: $_";
      my @vals = split;
      print $sock "!$vals[0] = 7 : not implemented ;\n";
    }
  }
  close($sock);

  $requested_rate = undef;
}


sub usage {
  print<<EOF;
Usage:
 mk5emu.pl -recorder <recorder>
EOF
  exit(1);
}

sub server_comm {
  return "OK" if ($debug);
  my ($type, $message) = @_;

  # Connect to the recorder server
  my $socket = IO::Socket::INET->new(PeerAddr => $recordername,
				     PeerPort => RECORDER_SERVER,
				     )
    || die "Could not connect to $recordername\n";


  #print "SENDING: <$type>$message</$type>";
  print $socket "<$type>$message</$type>";

  # Get response
  
  my $ret = "";
  while(<$socket>){
    $ret .= $_;
  }
  close($socket);


  if ($ret =~ /<fail>(.*)<\/fail>/s) {
    carp "$1";
    return undef;
  } elsif ($ret =~ /<succ \/>/s) {
    return "";
  } elsif ($ret =~ /<status>.*<\/status>/s) {
    return $1;
  } else {
    warn "Did not understand server reponse ($ret): $!\n";
    return undef;
  }
}


sub send_data($) {
  return server_comm('data', shift);
}

sub send_cmd($) {
  return server_comm('cmnd', shift);
}
