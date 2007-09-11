package PATH ;
use strict ;
use Data::Dumper ;
use Storable ;

use Time::Local;
use POSIX qw(strftime);




use vars qw(@ISA @EXPORT $VERSION);
require Exporter;
@ISA = qw(Exporter);
@EXPORT = qw(classpath dyld_library_path path ld_library_path manpath check cmtdeps chkdeps chkmtime);

=cut

=cut


sub present_var{ 

    for my $v (@_){
       &check($ENV{$v} ); 
    }
}

sub classpath         { &check($ENV{'CLASSPATH'} ); }
sub dyld_library_path { &check($ENV{'DYLD_LIBRARY_PATH'} ); }
sub ld_library_path   { &check($ENV{'LD_LIBRARY_PATH'} ); }
sub path              { &check($ENV{'PATH'} ); }
sub manpath           { &check($ENV{'MANPATH'} ); }


sub check {
  my ( $var ) = @_ ;

  printf "SCB::Workflow::PATH checking[%s]\n",$var ;
  my @elem = split /:/,$var ;

  for my $e (@elem){
	my $s = "n" ;
	$s = "d" if (-d $e );
	$s = "f" if (-f $e );
	
    printf "[%s]%s\n",$s,$e ;
  }

}



sub chkmtime {

 print "chkmtime.... \n"; 

 my @fs = ();
  while(<>){
	 if(m/^\s*(.*)\s*$/){
        push(@fs , $_) for(split(/ /,$1));
	 } else {
		 print "nomatch .... ",$_ ;
	 }
  }


 for my $f (@fs){
	  my $m = &mtime($f) ;
     printf "====[%-90s][%d][%s]\n",$f,$m,&fmt($m);
 }
}


sub chkdeps {

  my $rootsys = $ENV{'ROOTSYS'} ;
  my $obj = "" ;
  my $depl = "" ;
  my @deps = ();
  while(<>){
	 if(m/^(\S*):\s*(.*)\\/){
        $obj=$1;
		$depl.=$2 ;
	 } elsif ( m/\s*(.*)\s*\\/ ){
        $depl.=$1 ;
	 } elsif ( m/\s*(.*)\s*$/ ){
        $depl.=$1 ;
	 }
  }

  push(@deps, $_) for(split(/ /,$depl));
  #printf "obj %s %d \n",$obj,$#deps + 1  ;

  my $x = {} ; 
  #printf "$_ \n" for(@deps);
  for my $d (@deps){
	if($d =~ m|^/usr/lib| || $d =~ m|^/usr/include| || $d =~ m|^$rootsys| ){
	} else {
	  push(@{ $x->{$obj}{'deps'} }, $d ); 
	  my $m = &mtime($d) ;
	  push(@{ $x->{$obj}{'tims'} }, &fmt($m) ); 
	}		
  }
  print Dumper($x);

}


sub mtime {
 my ($path ) = @_ ;
 ## ($dev,$ino,$mode,$nlink,$uid,$gid,$rdev,$size, $atime,$mtime,$ctime,$blksize,$blocks) = stat($filename);
  my @a = -f $path ? stat($path)  : ();
  return $#a > 10 ? $a[9] : 0 ;
}


sub fmt{
 my( $time, $pfmt ) = @_ ;
 my $fmt = ( defined $pfmt )? $pfmt :  "%Y%m%d-%H%M%S" ;
 
 my $fdate;
 my @atime;
 if($time > 0){
    @atime= localtime($time);
    $fdate = strftime($fmt, @atime);
  } else {
    @atime= localtime();
    $fdate = strftime($fmt, @atime);
  }
 return $fdate;
} ## fmttime


sub cmtdeps {

=cut
     cd $DYW/DataStructure/MCEvent/$CMTCONFIG ;  cmtdeps MCEvent_dependencies.make

=cut


  my ( $file ) = @_ ;
  print "cmtdeps $file \n";  

  my $rootsys = $ENV{'ROOTSYS'} ;
  my $x = {} ;

  open(F , "<$file" ) || die "cannot open $file \n";
  while(<F>){
     my $line = $_ ;
	 if( $line =~ m/^(\S*)\s=\s(.*)/ ){
         my $obj  = $1 ;
		 my $deps = $2 ;
		 my @dep = split / /, $deps ;
		 printf "obj:[%s][%d]\n", $obj,$#dep+1 ;

		 for my $d (@dep){

             ## skip the system and root headers
			 if($d =~ m|^/usr/lib| || $d =~ m|^/usr/include| || $d =~ m|^$rootsys| ){
			 } else {
		        push(@{ $x->{$obj}{'deps'} }, $d ); 
			 }	
		 }	
	 } else {
		 print "ERROR no match \n" ;
	 }
  }
  close F;

  print Dumper($x);

}




1;

