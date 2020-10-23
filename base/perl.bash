perl-vi(){ vi $BASH_SOURCE ; } 
perl-oneliners(){ cat << EOU


Simple dump key value pairs, eg from simple json files
--------------------------------------------------------

::

   perl -ne 'm/\"(\S*)\"\s*:\s*\"(\S*)\"/ && print "$1 : $2\n" ' manifest.json

   perl -ne 'm/\"(type)\"\s*:\s*\"(\S*)\"/ && print "$2\n" ' manifest.json







EOU
}


##  NB for cpan config
##       vi .cpan/CPAN/MyConfig.pm
## echo $PATH | grep $CPAN_HOME/usr/local/bin || export PATH=$CPAN_HOME/usr/local/bin:$PATH
#export PERL_HOME=/Library/Perl/5.8.6


perl-env(){
   local msg="=== $FUNCNAME :"

}


perl-libs(){

   if [ "$NODE" == "g4pb" ]; then

      export X_CPAN_HOME=/opt/perl/cpan
      export CPAN_BUILD=/opt/perl/.cpan/build/CPAN-1.8801
      export MY_LIBS=/work/blyth/hfag/perl:/Users/blyth/perl
      export OSX_LIBS=$PERL_HOME:$PERL_HOME/darwin-thread-multi-2level
     #export CPAN_LIBS=$X_CPAN_HOME/lib/darwin-thread-multi-2level:$X_CPAN_HOME/lib:$CPAN_BUILD/lib
      export CPAN_LIBS=$X_CPAN_HOME/lib:$CPAN_BUILD/lib

     #export PERL5LIB=$MY_LIBS:$OSX_LIBS
     #export PERL5LIB=$MY_LIBS:$OSX_LIBS:$CPAN_LIBS
      export PERL5LIB=$MY_LIBS:$CPAN_LIBS

   else

      export PERL5LIB=$HOME/perl
   fi
}
   

perl-x(){ scp $HOME/$BASE_BASE/perl.bash ${1:-$TARGET_TAG}:$BASE_BASE ; }
perl-i(){ . $HOME/$BASE_BASE/perl.bash ; }

perl-x-pkg(){
	X=${1:-$TARGET_TAG}
	if ( [ "$NODE" == "g4pb" ] && [ "$X" != "G" ] ) ; then 
	   ssh $X "mkdir -p perl/SCB/Workflow" 
	   scp $HOME/$BASE_BASE/perl.bash $X:$BASE_BASE
	   scp $HOME/perl/SCB/Workflow/PATH.pm $X:perl/SCB/Workflow/ 
	else
	   echo cannot perl-x-pkg  from node $NODE unless it is g4pb , and cannot target $X node G	
	fi 
}

perl-strftime(){
  perl -MPOSIX -e  'print strftime( "%Y%m%d-%H%M%S" , localtime($ARGV[0]) );' ${1:-0} 
}



perl-alias(){

   ## this gives the same results as util:md5 in eXist XQuery 
   alias md5="perl -MDigest::MD5 -e 'use Digest::MD5 q(md5_hex); printf \"md5_hex[%s]=%s\n\", \$_,md5_hex(\$_) for(@ARGV) '" 

   alias ccp="perl -MSCB::Workflow::PATH -e '&classpath;' "
   alias dlp="perl -MSCB::Workflow::PATH -e '&dyld_library_path;' "
   alias ldp="perl -MSCB::Workflow::PATH -e '&ld_library_path;' "
   alias mp="perl -MSCB::Workflow::PATH -e '&manpath;' "
   alias pp="perl -MSCB::Workflow::PATH -e '&path;' "
   alias cmtdeps="perl -MSCB::Workflow::PATH -e '&cmtdeps(@ARGV);' "
   alias chkdeps="perl -MSCB::Workflow::PATH -e '&chkdeps(@ARGV);' "
   #alias chkmtime="perl -MSCB::Workflow::PATH -e '&chkmtime(@ARGV);' "
   alias chkmtime="PERL5LIB=$HOME/$BASE_BASE perl -MPATH -e '&chkmtime(@ARGV);' "
   alias ftime="perl -MPOSIX -e  'print strftime( '%Y%m%d-%H%M%S' , localtime($1) );'" 

}


