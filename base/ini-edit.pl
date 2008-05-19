#!/usr/bin/env perl -w
#
# the below works on Tiger and Linux
#    require "$ENV{'ENV_HOME'}/base/INI.pm" ;
#
# BUT stock perl 5.8.1 on Leopard is fussy regards environment in requires ???
#
my $dir = `dirname $0`;
chop $dir ;
#print "dollar0... $0 dir $dir \n";
 
require "$dir/INI.pm" ;
&INI::EDIT(@ARGV) ; 

