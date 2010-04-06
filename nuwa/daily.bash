# === func-gen- : nuwa/daily fgp nuwa/daily.bash fgn daily fgh nuwa
daily-src(){      echo nuwa/daily.bash ; }
daily-source(){   echo ${BASH_SOURCE:-$(env-home)/$(daily-src)} ; }
daily-vi(){       vim $(daily-source) ; }
daily-env(){      elocal- ; }
daily-usage(){
  cat << EOU
     daily-src : $(daily-src)
     daily-dir : $(daily-dir)


     daily-rev-    (no frills version)
     daily-rev
         Access the revision of the last successful dybinst autobuild 
         since the cutoff time 
         (currently cutoff times are daily at 18:00 dayabay time) 

         For a table of prior such revisions, times and links to build 
         status pages see
             http://dayabay.ihep.ac.cn/tracs/dybsvn/daily/dybinst

     daily-build-
         Builds specific revisions in folders such $(daily-revdir 9999) 
         beneath $(daily-dir). Revisions obtained using daily-rev-
         
     daily-build
         Calls daily-build- for the actual work, this creates day links
         such as $(daily-daydir) that point to the actual revdir $(daily-revdir 9999)
         in which the build is performed   
         
     daily-purge
         Deletes older build dirs and deletes the daylinks that point to them

EOU
}
daily-dir(){
  case $(hostname) in 
     lxslc??.ihep.ac.cn) echo /home/dyb/dybsw/NuWa/daily ;; 
        farm1.dyb.local) echo /home/dyb/dybsw/NuWa/daily ;; 
                      *) echo /tmp/env/$FUNCNAME         ;;
  esac 
}
daily-cd(){  cd $(daily-dir); }
daily-tcd(){  cd $(daily-todaydir); }
daily-mate(){ mate $(daily-dir) ; }
daily-get(){
   local dir=$(dirname $(daily-dir)) &&  mkdir -p $dir && cd $dir
}

daily-creds(){ echo "dayabay:$(<~/.dybpass)" ; }
daily-url(){   echo "http://dayabay.ihep.ac.cn/tracs/dybsvn/daily/dybinst?format=txt" ; }
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
   local rev
   rev=$(daily-rev-)
   rc=$?
   [ "$rc" != "0" ] && echo $msg ABORT failed to obtain revision rc $rc && return 1    

   local ddir=$(daily-dir)
   mkdir -p $ddir && cd $ddir
   
   daily-build- $rev
   rc=$?
   [ "$rc" != "0" ] && echo $msg ABORT && return $rc     

   daily-cd
   
   local bdir=$(daily-revdir $rev)
   ln -sf $bdir $(daily-daydir)
   touch $bdir/days
   echo $(daily-daydir) >> $bdir/days

}

daily-dybinst-url(){     echo http://dayabay.ihep.ac.cn/svn/dybsvn/installation/trunk/dybinst/dybinst ; }
daily-build-(){
  local msg="=== $FUNCNAME :"
  local rev=$1
  local rdir=$(daily-revdir $rev)
  [ -z "$rev" ]  && echo $msg revision argument required && return 1 
  # [ -d "$rdir" ] && echo $msg revdir $rdir exists already && return 0 

  echo $msg proceeding with build $rdir ...
  local iwd=$PWD
  mkdir -p $rdir && cd $rdir

  [ -f "dybinst" ] && echo $msg dybinst already exported || svn export $(daily-dybinst-url)

  local cmd
  local rc
  cmd="./dybinst -z $rev -e ../../external trunk checkout"
  echo $msg $cmd
  eval $cmd
  rc=$?
  [ "$rc" != "0" ] && echo $msg ABORT checkout failed rc:$rc && return $rc     

 
  cmd="./dybinst -z $rev -e ../../external -c trunk projects"
  echo $msg $cmd
  eval $cmd
  rc=$?
  [ "$rc" != "0" ] && echo $msg ABORT projects failed rc:$rc && return $rc     


  cd $iwd
}



daily-validate(){
  local rev=$1
  local rdir=$(daily-revdir $1) 

  . $rdir/installation/trunk/dybtest/scripts/dyb__.sh   
  cd $rdir/NuWa-trunk
  daily-validate-
}


daily-validate-(){
  local msg="=== $FUNCNAME :"
  local iwd=$PWD

  cd $iwd/dybgaudi
  dyb__testall -v

  cd $iwd/tutorial
  dyb__testall -v

  local rc
  cd $iwd
  dyb__testall_ok
  rc=$?

  cd $iwd
}




daily-keep(){  echo 7 ; }
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

