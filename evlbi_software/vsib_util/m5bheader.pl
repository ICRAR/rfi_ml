#!/usr/bin/perl -w

use strict;

my ($buf, $nread, $tvg);
my ($sync, $user, $frame, $t1, $t2, $crc);
my ($jjj, $sss);

sub initialize_crc_table($);
sub crc ($$);

my @crc_table = ();

initialize_crc_table(040003);

foreach (@ARGV) {
  open(M5B, $_) || die "Could not open $_: $!\n";

  print "Reading $_\n\n";
  
  my $gotsome = 0;
  while (1) {
    $nread = sysread(M5B, $buf, 16);
    if (! defined($nread)) {
      printf("Error reading $_: $!");
      last;
    } elsif ($nread==0) { # EOF
      print "Warning: No data read\n" if (!$gotsome);
      last;
    } elsif ($nread!=16) {
      print "Warn: Only read $nread bytes. Stopping\n";
      last;
    }
    $gotsome = 1;

    ($sync, $frame, $user, $t1, $crc, $t2) = unpack 'LSSLSS', $buf;

    print "-------------------\n";
    printf "Sync:  %x\n", $sync;
    printf "User:  %x\n", $user;
    
    if ($frame & 0x8000) {
      $tvg = 'Yes';
    } else {
      $tvg = 'No';
    }
    print "TVG:   $tvg\n";
    $frame &= 0x7FFF;
    printf "Frame: %d\n", $frame;

    $jjj = $t1>>28 & 0xF;
    $jjj .= $t1>>24 & 0xF;
    $jjj .= $t1>>20 & 0xF;
    $sss = $t1>>16 & 0xF;
    $sss .= $t1>>12 & 0xF;
    $sss .= $t1>>8 & 0xF;
    $sss .= $t1>>4 & 0xF;
    $sss .= $t1 & 0xF;
    $sss .= ".";
    $sss .= $t2>>12 & 0xF;
    $sss .= $t2>>8 & 0xF;
    $sss .= $t2>>4 & 0xF;
    $sss .= $t2 & 0xF;
    print "Time:  $jjj/$sss\n";

    printf "CRC:   %04X\n", $crc;

    $crc = crc($t1, $t2);

    printf ("  : %04X\n", $crc);

    sysseek(M5B, 10000, 1);
  }


  close($_);
}


sub initialize_crc_table($) {
  my ($key) = @_;

  for (my $i = 0; $i < 256; $i++) {
    my $reg = $i << 8;

    for (my $j = 0; $j < 8; $j++) {
      $reg <<= 1;
      if ($reg & 0x10000) {
	$reg ^= $key;
     }
   }
    $crc_table[$i] = $reg&0xFFFF;
  }
}



sub crc ($$) {
  my ($t1, $t2) = @_;

  my @bytes = ();
  for (my $i=24; $i>=0; $i -= 8) {
    push @bytes, ($t1>>$i)&0xFF;
  }
  for (my $i=8; $i>=0; $i -= 8) {
    push @bytes, ($t2>>$i)&0xFF;
  }

  my $crc = 0;
  foreach (@bytes) {
    my $top = ($crc>>8);
    
    #print "TOP: $top\n";
    #print "CRC: $crc\n";
    #print "BYTE: $_\n";
    #print "CRC_TABLE[$top]: ", $crc_table[$top], "\n";
    #print "crc<<8: ", $crc<<8, "\n";
    #print "(crc<<8)&0xFFFF+\$_: ", (($crc<<8)&0xFFFF)+$_, "\n";

    $crc = ((($crc<<8)&0xFFFF)+$_)^$crc_table[$top];

    #printf "%0x : %x\n", $crc, $_;
  }
  return $crc;
}

