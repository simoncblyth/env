#!/usr/bin/env perl -w

=cut
     add "remove" support 
     add required ordering support , after smth else 
=cut


use Data::Dumper;

my $conf   = $ARGV[0] ||  "/tmp/httpd.conf" ;
my $name   = $ARGV[1] || "last"  ;  ## name eg "python"
my $op     = $ARGV[2] || "list" ;  ## add remove list


&read_modules( $conf );
my $m = &module_hash( $name );
my $l = &module_hash( "last" );

print Dumper( $m );

if( $op eq "add" ){
  unless( defined $m ){
     &add_line( $conf , $l->{'line'} , &module_line( $name ) );  
  } else {
	  print "cannot add $name as already present \n ";
  }
} else {
  print "op $op not handled \n";

}


sub add_line(){
   	
   my ( $file , $num  , $line ) = @_ ;
   my @lines = ();
   push(@lines, $line );
   
   my $patch = &addline_patch( $num , \@lines );  
   my $pfile = "/tmp/patch$$" ;
   open(P, ">$pfile" ) || die "cannot open $pfile " ; 
   print P $patch ;
   close P ;
   my $cmd = "cat $pfile ; \$ASUDO patch -b $file <  $pfile ; diff $file\{.orig,\} " ;
   print "$cmd\n" ;
   print `$cmd` ;
}


sub module_line(){
  my ( $name ) = @_ ;
  return sprintf("LoadModule %s_module libexec/mod_%s.so", $name,$name );
}


sub module_hash(){
	
  my ( $name ) = @_ ; 
  my $m ;
  if( defined $main::index->{$name} ){
     $m = $main::index->{$name};
  } elsif ( $name eq "last" ){
     $m = ${ $main::mods }[-1] ;
  }
  return $m ;

}


sub read_modules(){
  my ( $conf ) = @_ ; 
  my $module ;
  my $name ;
  my @mods=();
  open(C, "<$conf" ) || die "cannot open $conf\n " ;
  while(<C>){
    if(m/^(LoadModule\s*(\S*)_module\s*(\S*))\n$/ ){
		$name = $2 ;
	    $module = { 'text'=>$1 , 'line'=>$. , 'module'=>$name , 'so'=>$3 } ;
	    push(@{ $main::mods }, $module );
		$main::index->{$name} = $module ;
    }
  }
  close C;
}




sub addline_patch(){
  #	
  # simulating the output of diff 	
  #
  my ( $num , $radd ) = @_ ;        ## $num is the linenumber up to which they are the same 
  my $ret = sprintf("%da%d\n", $num  , $#{ $radd  } + 1 + $num   );
  $ret.="> $_\n" for(@{ $radd });
  return $ret ;
}







