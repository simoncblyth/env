#!/usr/bin/env perl -w

use Data::Dumper ; 
use INI ;

my ( $file , @edits ) = @ARGV ;

$c = new INI ;
$c->read( $file );
$c->edit( @edits );
$c->prepare();
$c->write("$file.out" );


print Dumper( $c);

