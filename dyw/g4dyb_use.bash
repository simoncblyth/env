#
#
#    g4dyb-use-sub
#    g4dyb-use
#
#    issue :
#       G4dyb is emitting control codes.. 0x1B (escapes) that are not valid in xml
#
#       attempt to remove them with tr fail :
#
#         tr -d [:cntrl:]  g4dyb.out   ... removes the newlines too !!
#         cat g4dyb.out |  tr -d "[:cntrl:]"    removes the newlines too
#
#         cat g4dyb.out | native2ascii > t.xml 
#         xmllint -noout t.xml     
#               still not valid xml
#
#
#     perl -n -e 'm/^\e\[0m(.*)\e\[0m(.*)$/ && print "$1 $2\n" '  g4dyb.out
#
#     perl -pe 's/\e\[0m//g' g4dyb.out    makes the terminal red !!
#
#    succeeds to kill the control codes 
#        perl -i.orig -pe 's/\e\[..//g' g4dyb.out
#  
#
#
#
#
#   so remove 
#    .AddFormat( new FmtLevelColor() )   from LogG4dyb.cc
#
#   results in valid
#      xmllint -noout g4dyb.out
#
#    add an alias to httpd.conf 
#   [blyth@grid1 condor]$ chmod -R o+r g4dyb
#
#    http://grid1.phys.ntu.edu.tw:8080/g4dyb/
#
#
#   Alias /g4dyb/   /disk/d4/blyth/condor/g4dyb/
#
#
#   http://root.cern.ch/root/WebFile.html
#
#    directory layout
#
#   /disk/d4/blyth/g4dyb/electron
#   /disk/d4/dywlog/thho
#
#
[ "$DYW_DBG" == "1" ] && echo $DYW_BASE/g4dyb_use.bash start 

g4dyb-use-x(){ scp $HOME/$DYW_BASE/g4dyb_use.bash ${1:-$TARGET_TAG}:$DYW_BASE; }
g4dyb-use-i(){ .   $HOME/$DYW_BASE/g4dyb_use.bash ; }



g4dyb-use-branch(){

  local release=${DYW_VERSION%_wc}
  local fold=""   ## "G4dyb"
  local branch=${1:-dummy}
  shift
  
  svn-branch $release $fold $branch "$*"
}


g4dyb-use-sub(){

  func=g4dyb-use
  macro=${1:-$DEFAULT_MACRO} 

  [ "X$macro" == "X" ] && echo need macro name as argument && return
  path=jobs/$func/$macro

  echo checking version of working copy $DYW before submission 
  version=$(svnversion $DYW)
  clean=$(echo $version | perl -n -e 'print m/^\d*$/?1:0;' )
  
  ## creates timestamped folder and submits  NB this is not condor_submit 

  if [ "$clean" == "1" ]; then 
    batch-submit $path $func "$@"
  else
	cat << EOM  

	g4dyb-sub submission is DISALLOWED  as the repository does not have a clean version number $version 
	clean the repository by commiting (resolving any conflicts) and then updating 
	    cd \$DYW 
		svn commit 
		svn update 
EOM

  fi
  

}




g4dyb-use(){    

    [ "$(pwd)" == "$DYM" ] && ( echo running from the macro folder $DYM is prohibited && return )
   
       func=g4dyb-use
	exename=G4dybApp
  macroname=${1:-$DEFAULT_MACRO}
     shift

   version=$(svnversion $DYW)
   clean=$(echo $version | perl -n -e 'print m/^\d*$/?1:0;' )
   scmtag=$(basename ${DYW%_wc})   ## strip the _wc from the working copy directory 
   tracurl=$(scm-use-tracurl $scmtag $version) 

   printf "<scm tag=\"%s\" rev=\"%s\" host=\"%s\" port=\"%s\" >\n" $scmtag $version $SCM_HOST $SCM_PORT
   printf "<trac>\n"
   printf "<url>%s</url>\n" $tracurl
   printf "</trac>\n"
   printf "<version clean=\"%d\" >%s</version>\n" $clean $version  
   svn info --xml $DYW | perl -ne '$.==1||print'    ## excluding the 1st <?xml?> line
   printf "</scm>\n"

     args="$@"

     runmac=$macroname.mac
     srcmac=$DYM/$runmac
	  setup=$DYW/G4dyb/cmt/setup.sh


    ## copy the src macro and do minor editing to for this run
	
    cp $srcmac $runmac
	perl -pi -e 's|^#\s*(/run/beamOn)\s*(\d*)\s*$|$1 5|' $runmac

    ## set up the CMT controlled runtime environment , including PATH to find the executable
    ## all the arguments are being swalloed by the setup script somehow ??
	## this removes the positional arguments , as the CMT setup script uses a $* as arguments to cmt 
    ##  resulting in cmt getting the argumenst inadvertantly
    ##       http://tldp.org/LDP/abs/html/internalvariables.html#INCOMPAT
	set -- 
	
    printf "<setup>\n" 
	xml-cdata-open
	. $setup
	xml-cdata-close
    printf "</setup>\n" 
	
	exe=$(which $exename.exe)

   ##  redirect stdout and stderr from the executable for more control
   ##  and to allow post-processing of the output  
   ##     http://tldp.org/HOWTO/Bash-Prog-Intro-HOWTO-3.htmlA
   ##     ( echo hello && ls -1 asjhdxa ) 1> /tmp/out 2> /tmp/err
   ##
	cmd=" $exe $runmac $args  " 



    printf "<%s exename=\"%s\" macroname=\"%s\"  >\n" $func  $exename $macroname 
	
    xml-stringlist-element cmd "$cmd"
    xml-file-element   runmacro $runmac 
 
    printf "<input>\n" 
    xml-path-element setup      $setup 
    xml-path-element executable $exe
    xml-path-element srcmacro   $srcmac
    xml-path-element runmacro   $runmac
    printf "</input>\n" 

    xml-llp-element
    xml-env-element 
	xml-ldd-element $exe
   
    ## is it advisable to keep the stdout in here ???
    ## ... I guess so, as event numbers will not be huge ... so keeping it
	##  together is ok

    printf "<stdout>\n" 
	xml-cdata-open
	#( eval "$cmd" )  1>$func.exe.out 2>$func.exe.err
	#$cmd  1>$func.exe.out 2>$func.exe.err
	eval $cmd 
	xml-cdata-close
    printf "</stdout>\n" 
	
    printf "</%s>\n" $func

}

[ "$DYW_DBG" == "1" ] && echo $DYW_BASE/g4dyb_use.bash finished 

