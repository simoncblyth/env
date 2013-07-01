#!/usr/bin/env perl

print "demo.pl hello\n";

while (@ARGV) {
   print "ARG: $ARGV[0] \n";
   shift @ARGV;
}



