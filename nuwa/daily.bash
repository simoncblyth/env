# === func-gen- : nuwa/daily fgp nuwa/daily.bash fgn daily fgh nuwa
daily-src(){      echo nuwa/daily.bash ; }
daily-source(){   echo ${BASH_SOURCE:-$(env-home)/$(daily-src)} ; }
daily-vi(){       vim $(daily-source) ; }
daily-env(){      elocal- ; }
daily-usage(){
  cat << EOU


     DEVELOPMENT HERE IS DEPRECATED ....
        REHOMED INTO dybsvn 
            http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/installation/trunk/dybtest/scripts/daily.bash

        Documentation pointers at 
            https://wiki.bnl.gov/dayabay/index.php?title=Offline_Software_Installation#Setup_of_daily_builds_and_validations


  == Function Descriptions ==

     daily-src : $(daily-src)
     daily-dir : $(daily-dir)

     daily-url : $(daily-url)
     daily-rev-    
         (no frills version)
         Looks up the revision ; requires the existance of a protected
         file $HOME/.dybpass , create this file with 
              echo youknowwhat > $HOME/.dybpass

     daily-rev
         Access the revision of the last successful dybinst autobuild 
         since the cutoff time 
         (currently cutoff times are daily at 18:00 dayabay time) 

         For a table of prior such revisions, times and links to build 
         status pages see
             http://dayabay.ihep.ac.cn/tracs/dybsvn/daily/dybinst

     daily-dir : $(daily-dir)
         the directory within which revdir (eg NuWa-9999) are created
         holding the revision builds, typically set to \$SITEROOT/../daily 

     daily-build-
         Builds specific revisions in folders such $(daily-revdir 9999) 
         beneath $(daily-dir). Revisions obtained using daily-rev-
         
     daily-build
         Calls daily-build- for the actual work, this creates day links
         such as $(daily-daydir) that point to the actual revdir $(daily-revdir 9999)
         in which the build is performed   
         
     daily-purge
         Deletes older build dirs and deletes the daylinks that point to them



     daily-rc  : $(daily-rc)
          path to the configuration file

     daily-cfg
          list the settings obtained from the configuration file
     daily-cfg-check
          check contents of config file or create a demo one if not existing with -init
     daily-cfg-init
         initialize the config using daily-demorc-
     daily-cfg-demo
         emit demonstration config to stdout



EOU
}

daily-cfg-demo(){  cat << EOD
#
# created $(date) by $FUNCNAME
#
# credentials used to access the Trac master to determine the revisions to build
#
local creds=dayabay:youknowit

#
# absolute path to "daily" dir within which revdir such as NuWa-9999 
# will be created, eg: /home/dyb/dybsw/NuWa/daily or /data/env/local/dyb/trunk/daily
#
# dybinst will be exported into the revdir and run 
# positioning "daily" inside an pre-existing "dybinst" 
# containing directory allows the external dir to be shared 
# using relative path ../../external
#
local dir=/path/to/daily

# relative (or absolute) path from revdir (eg /path/to/daily/NuWa-9999) 
# to the external directory. 
# Used as the "-e" dybinst argument from the revdir 
#
local external=../../external

#
# number of revdir to be retained by the daily-purge function 
#
local keep=7


EOD
}

daily-rc(){       echo $HOME/.dyb__dailyrc ; }
daily-edit(){     vi $(daily-rc) ; }

daily-external(){ [ -f "$(daily-rc)" ] && . $(daily-rc) ; echo $external ;  }
daily-creds(){    [ -f "$(daily-rc)" ] && . $(daily-rc) ; echo $creds    ;  }
daily-dir(){      [ -f "$(daily-rc)" ] && . $(daily-rc) ; echo $dir      ;  }
daily-keep(){     [ -f "$(daily-rc)" ] && . $(daily-rc) ; echo $keep     ;  }

daily-cfg(){  cat << EOS
Configuration read from daily-rc yields the below settings :
   daily-rc       : $(daily-rc)
  
   daily-external : $(daily-external)
   daily-creds    : $(daily-creds)
   daily-dir      : $(daily-dir)
   daily-keep     : $(daily-keep)

EOS
}
daily-cfg-check(){
  local msg="=== $FUNCNAME :"
  [ ! -f "$(daily-rc)" ] && echo $msg ERROR no daily-rc : $(daily-rc) attempting to create one ... && daily-cfg-init
  . $(daily-rc)
  [ -z "$creds" ]    && echo $msg daily \"creds\" undefined && return 1
  [ -z "$dir" ]      && echo $msg daily \"dir\" undefined && return 1
  [ -z "$external" ] && echo $msg daily \"external\" undefined && return 1
  [ -z "$keep" ]     && echo $msg daily \"keep\" undefined && return 1
  return 0
}
daily-cfg-init(){
  local msg="=== $FUNCNAME :"
  [ -f "$(daily-rc)" ] && echo $msg daily-rc : $(daily-rc) exists already && return 0
  echo $msg creating demo $(daily-rc) 
  daily-cfg-demo > $(daily-rc) 
  chmod go-rw $(daily-rc)
  echo $msg YOU NEED TO EDIT : $(daily-rc)
}




daily-cd(){  cd $(daily-dir); }
daily-tcd(){  cd $(daily-todaydir); }

daily-dybinst-url(){     echo http://dayabay.ihep.ac.cn/svn/dybsvn/installation/trunk/dybinst/dybinst ; }
daily-url(){             echo "http://dayabay.ihep.ac.cn/tracs/dybsvn/daily/dybinst?format=txt" ; }

daily-rev-(){   curl --fail -s -u "$(daily-creds)" "$(daily-url)" ; }
daily-rev(){
  local msg="=== $FUNCNAME :"  
  local rev
  local rc
  rev=$(daily-rev-)
  rc=$?
  ## cannot combine prior 4 lines to 2 and capture both output and rc 
  case $rc in
     0) echo $msg last revision is $rev ;;
     *) echo $msg FAILED with to access last revision ... curl gave non-zero rc $rc  ... bad credentials OR bad server config  ;;
  esac
}

daily-revdir(){   echo NuWa-$1 ; }
daily-daydir(){   echo NuWa-$(date +"%Y%m%d") ; }
daily-todaydir(){ echo $(daily-dir)/$(daily-daydir) ; }


daily-build(){
   local msg="=== $FUNCNAME :"
   local rev=${1:-$(daily-rev-)}
   [ -z "$rev" ]  && echo $msg revision is required && return 1 

   local ddir=$(daily-dir)
   mkdir -p $ddir && cd $ddir

   local rdir=$(daily-revdir $rev)
   # [ -d "$rdir" ] && echo $msg revdir $rdir exists already && return 0 

   echo $msg proceeding with build $rdir ...
   local iwd=$PWD
   mkdir -p $rdir && cd $rdir

   daily-build- 
   rc=$?
   [ "$rc" != "0" ] && echo $msg daily-build- ABORT    && return $rc     

   daily-validate- 
   rc=$?
   [ "$rc" != "0" ] && echo $msg daily-validate- ABORT && return $rc     


   ## publishing via the day link is only done if the above build + validation succeeds
   daily-cd
   ln -sf $rdir $(daily-daydir)
   touch $rdir/days
   echo $(daily-daydir) >> $rdir/days
}



daily-build-(){
  local msg="=== $FUNCNAME :"
  [ -f "dybinst" ] && echo $msg dybinst already exported || svn export $(daily-dybinst-url)
  local cmd
  local rc
  cmd="./dybinst -z $rev -e $(daily-external) trunk checkout"
  echo $msg $cmd
  eval $cmd
  rc=$?
  [ "$rc" != "0" ] && echo $msg ABORT checkout failed rc:$rc && return $rc     
  cmd="./dybinst -z $rev -e $(daily-external) -c trunk projects"
  echo $msg $cmd
  eval $cmd
  rc=$?
  [ "$rc" != "0" ] && echo $msg ABORT projects failed rc:$rc && return $rc     
  cd $iwd
}

daily-validate-(){
  local iwd=$PWD
  local rc
  [ ! -f "dybinst" ]      && echo $msg ERROR revdir MUST contain dybinst      && return 1 
  [ ! -d "installation" ] && echo $msg ERROR revdir MUST contain installation && return 1 
  [ ! -d "NuWa-trunk" ]   && echo $msg ERROR revdir MUST contain NuWa-trunk   && return 1 
  . ./installation/trunk/dybtest/scripts/dyb__.sh   
  dyb__validate 
  rc=$?
  cd $iwd
  return $rc
}

daily-purge(){
  local msg="=== $FUNCNAME :"
  local nmax=$(daily-keep)
  local revs
  local irev
  local nrev
  declare -a revs
  echo $msg   
  daily-cd
  revs=($(find . -maxdepth 1 -name 'NuWa-????' | sort ))
  nrev=${#revs[@]}
  echo $msg pwd:$PWD nrev:$nrev nmax:$nmax
  iday=0
  local cmd
  while [ "$irev" -lt "$nrev" ]
  do
    local rev=${revs[$irev]}
    if [ $(( $nrev - $irev > $nmax )) == 1 ]; then
        ## tidy up day links that point to the folder to be deleted
        local daylink  
        cat $rev/days | while read daylink ; do
            local revd=$(readlink $daylink)
            echo $msg daylink $daylink $revd
            if [ "$revd" == "$(basename $rev)" ]; then
               cmd="rm $daylink"
               echo $msg remove daylink $daylink ... $cmd
               #eval $cmd       
            else
                echo revd $revd rev $rev
            fi  
        done      
        cmd="rm -rf $rev"
        echo $msg delete $rev ... $cmd
        #eval $cmd
    else
        echo retain $rev
    fi
    let "irev = $irev + 1"
  done
}

