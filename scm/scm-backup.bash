scm-backup-src(){ echo scm/scm-backup.bash ; }
scm-backup-source(){  echo $(env-home)/$(scm-backup-src) ; }
scm-backup-url(){     echo $(env-url)/$(scm-backup-src) ; }
scm-backup-vi(){      vi $(scm-backup-source) ; }


scm-backup-log(){  cat << EOL

2012 May 7th removing cms02 native inhibiters

	[root@cms02 repos]# cd /var/scm/repos
	[root@cms02 repos]# ll *-scm-recover-repo
	lrwxrwxrwx  1 root root 75 May  4 19:05 aberdeen-scm-recover-repo -> /var/scm/backup/cms02/repos/aberdeen/2012/05/01/130104/aberdeen-1599.tar.gz
	lrwxrwxrwx  1 root root 65 May  4 19:05 data-scm-recover-repo -> /var/scm/backup/cms02/repos/data/2012/05/01/130104/data-23.tar.gz
	lrwxrwxrwx  1 root root 65 May  4 19:05 env-scm-recover-repo -> /var/scm/backup/cms02/repos/env/2012/05/01/130104/env-3443.tar.gz
	lrwxrwxrwx  1 root root 70 May  4 19:05 heprez-scm-recover-repo -> /var/scm/backup/cms02/repos/heprez/2012/05/01/130104/heprez-764.tar.gz
	lrwxrwxrwx  1 root root 71 May  4 19:05 newtest-scm-recover-repo -> /var/scm/backup/cms02/repos/newtest/2012/05/01/130104/newtest-20.tar.gz
	lrwxrwxrwx  1 root root 72 May  4 19:05 tracdev-scm-recover-repo -> /var/scm/backup/cms02/repos/tracdev/2012/05/01/130104/tracdev-123.tar.gz
	[root@cms02 repos]# rm *-scm-recover-repo 

	[root@cms02 tracs]# ll *-scm-recover-repo
	lrwxrwxrwx  1 root root 70 May  4 19:05 aberdeen-scm-recover-repo -> /var/scm/backup/cms02/tracs/aberdeen/2012/05/01/130104/aberdeen.tar.gz
	lrwxrwxrwx  1 root root 62 May  4 20:06 data-scm-recover-repo -> /var/scm/backup/cms02/tracs/data/2012/05/01/130104/data.tar.gz
	lrwxrwxrwx  1 root root 66 May  4 20:06 dybaux-scm-recover-repo -> /var/scm/backup/cms02/tracs/dybaux/2012/05/01/130104/dybaux.tar.gz
	lrwxrwxrwx  1 root root 66 May  4 20:09 dybsvn-scm-recover-repo -> /var/scm/backup/cms02/tracs/dybsvn/2011/10/17/183533/dybsvn.tar.gz
	lrwxrwxrwx  1 root root 60 May  4 20:10 env-scm-recover-repo -> /var/scm/backup/cms02/tracs/env/2012/05/01/130104/env.tar.gz
	lrwxrwxrwx  1 root root 66 May  4 20:12 heprez-scm-recover-repo -> /var/scm/backup/cms02/tracs/heprez/2012/05/01/130104/heprez.tar.gz
	lrwxrwxrwx  1 root root 68 May  4 20:13 newtest-scm-recover-repo -> /var/scm/backup/cms02/tracs/newtest/2012/05/01/130104/newtest.tar.gz
	lrwxrwxrwx  1 root root 66 May  4 20:13 toysvn-scm-recover-repo -> /var/scm/backup/cms02/tracs/toysvn/2012/05/01/130104/toysvn.tar.gz
	lrwxrwxrwx  1 root root 68 May  4 20:13 tracdev-scm-recover-repo -> /var/scm/backup/cms02/tracs/tracdev/2012/05/01/130104/tracdev.tar.gz
	[root@cms02 tracs]# 
	[root@cms02 tracs]# rm *-scm-recover-repo
	rm: remove symbolic link `aberdeen-scm-recover-repo'? y
	rm: remove symbolic link `data-scm-recover-repo'? n
	rm: remove symbolic link `dybaux-scm-recover-repo'? n
	rm: remove symbolic link `dybsvn-scm-recover-repo'? n
	rm: remove symbolic link `env-scm-recover-repo'? y
	rm: remove symbolic link `heprez-scm-recover-repo'? y
	rm: remove symbolic link `newtest-scm-recover-repo'? y
	rm: remove symbolic link `toysvn-scm-recover-repo'? n
	rm: remove symbolic link `tracdev-scm-recover-repo'? y
	[root@cms02 tracs]# 




EOL
}




scm-backup-usage(){
cat << EOU



   NON-STANDARD PORT

        -e "ssh -p $portNumber"


   STATE OF LOCK ADDITIONS

       * cannot incorp the scm-backup-rsync LOCKS in the IHEP->NTU transfers due to 
         lack of permissions : working with Q to incorp the scm-backup-rsync into the root cron task 
         by reviving the ssh-agent hook up 

   ABOUT LOCKING : GLOBAL AND RSYNC LOCKS

      * during scm-backup-all the "global" LOCKED directory $SCM_FOLD/LOCKED is created 
      * scm-backup-rsync pays attention to this LOCKED, and will abort if present
      * during scm-backup-rsync both the global LOCKED as described above and additional rsync LOCKED 
        are planted in eg $SCM_FOLD/backup/cms02/LOCKED/ during each transfer to partnered remote nodes
      * following rsync completion the rsync LOCKED is removed and a quick re-rsync is done to remove the LOCKED
    
     Note that the rsync LOCKED status is propagated to the remote directory during the rsync transfer, thus
     avoiding usage during transfers.

   INTEGRITY CHECKS

      Locking now prevents backup/rsync/recover functions both locally and remotely from touching partials.
      The backup procedures are purported to be hotcopy of themselves although mismatches 
      between what gets into the trac instance backup and the svn repo backup are possible.
      Such mismatches would not cause corruption however, probably just warnings from Trac syncing. 

      The DNA check ensures that the tarball content immediately after creation corresponds 
      precisely to the tarball at the other end of the transfers.

      * scm-backup-trac 
           * scm-tgzcheck-trac  : does a ztvf to /dev/null, extracts trac.db from tgz, dumps trac sql using sqlite3 
           * scm-backup-dna     : writes python dict containing md5 digest and size of tgz in sidecar .dna file
      * scm-backup-repo
           * scm-tgzcheck-ztvf  : does a ztvf to /dev/null
           * scm-backup-dna     : as above

      * scm-backup-rsync
           * performs remote DNA check for each paired backup node with scm-backup-dnachecktgzs : 
             finds .tar.gz.dna and looks for mutants (by comparing sidecar DNA with recomputed)
                
    DURING AN RSYNC TRANSFER, BOTH SIZE AND DIGEST DIFFER 

         [dayabay] /home/blyth/env > ~/e/base/digestpath.py  /home/scm/backup/dayabay/svn/dybaux/2011/10/19/100802/dybaux-5086.tar.gz
         {'dig': '7b87e78cc03ea544e2ad3abae46eecd1', 'size': 1915051630L}

         [blyth@cms01 ~]$  ~/e/base/digestpath.py  /data/var/scm/backup/dayabay/svn/dybaux/2011/10/18/100802/dybaux-5083.tar.gz
         {'dig': 'da39aee61a748602a15c98e3db25d008', 'size': 1915004348L}

         [blyth@cms01 ~]$  ~/e/base/digestpath.py  /data/var/scm/backup/dayabay/svn/dybaux/2011/10/18/100802/dybaux-5083.tar.gz
         {'dig': 'da39aee61a748602a15c98e3db25d008', 'size': 1915004348L}
                       sometime later, there is no change : transfer stalled  ?


   ISSUES WITH NEW INTEGRITY TESTS
       * SCM_BACKUP_TEST_FOLD  ignored by scm-backup-purge
       * expensive 
       * temporarily take loadsa disk space ~4GB (liable to cause non-interesting problems)

   IHEP CRON RUNNING OF THE BACKUPS ...
       changed Aug 2011 :  Cron jobs time changed to 15pm(Beijing Time) and 09am(beijing).


   POTENTIAL scm-backup-repo ISSUE AT 2GB 

      http://subversion.apache.org/faq.html#hotcopy-large-repos

           Early versions of APR on its 0.9 branch, which Apache 2.0.x and Subversion 1.x use, 
           have no support for copying large files (2Gb+). 
           A fix which solves the 'svnadmin hotcopy' problem has been applied and 
           is included in APR 0.9.5+ and Apache 2.0.50+. 
           The fix doesn't work on all platforms, but works on Linux.

      On C2 are using source apache  /data/env/system/apache/httpd-2.0.63 


   HOW TO RECOVER dayabay TARBALLS ONTO cms02, run from C2 (sudo is used)

      1) from C2  : scm-backup-rsync-dayabay-pull-from-cms01
      2) from C2  : scm-recover-all dayabay

    Note potential issue of incomplete tarballs, to reduce change 

   HOW TO TEST SOME IMPROVED ERROR CHECKING WITH SINGLE REPO/TRAC BACKUPS

     Run as root, eg from C2R::

         scm-backup-         ## pick up changes
         t scm-backup-repo   ## check the function   

         mkdir -p /tmp/bkp
         scm-backup-repo newtest /var/scm/repos/newtest /tmp/bkp dummystamp

         export LD_LIBRARY_PATH=/data/env/system/sqlite/sqlite-3.3.16/lib:$LD_LIBRARY_PATH   ## for the right sqlite, otherwise aborts
         scm-backup-trac newtest /var/scm/tracs/newtest /tmp/bkp dummystamp


   TESTING FULL BACKUP INTO TMP DIRECTORY

     Run as root, eg from C2R::

         scm-backup-
         t scm-backup-all   ## check the function   

         rm -rf /tmp/bkptest ; mkdir -p /tmp/bkptest
         export LD_LIBRARY_PATH=/data/env/system/sqlite/sqlite-3.3.16/lib:$LD_LIBRARY_PATH
         cd /tmp ; SCM_BACKUP_TEST_FOLD=/tmp/bkptest scm-backup-all


    SLIMMING THE TRAC TGZ ... ALL THOSE BITTEN LOGS

         http://bitten.edgewall.org/ticket/519

{{{
DELETE FROM bitten_log_message WHERE log IN (SELECT id FROM bitten_log WHERE build IN (SELECT id FROM bitten_build WHERE rev < 23000 AND config = 'trunk'))
DELETE FROM bitten_log WHERE build IN (SELECT id FROM bitten_build WHERE rev < 23000 AND config = 'trunk')
DELETE FROM bitten_error WHERE build IN (SELECT id FROM bitten_build WHERE rev < 23000 AND config = 'trunk')
DELETE FROM bitten_step WHERE build IN (SELECT id FROM bitten_build WHERE rev < 23000 AND config = 'trunk')
DELETE FROM bitten_slave WHERE build IN (SELECT id FROM bitten_build WHERE rev < 23000 AND config = 'trunk')
DELETE FROM bitten_build WHERE rev < 23000 AND config = 'trunk'
}}}





   \$SCM_FOLD   : $SCM_FOLD
   \$BACKUP_TAG : $BACKUP_TAG

   scm-backup-du   :   local  backup .gz sizes  in \$SCM_FOLD 
   scm-backup-rls  :   remote ls the .gz on the paired backup node $BACKUP_TAG
   scm-backup-mail :   send mail with the remote list  

   scm-backup-check :  find tarballs on all backup nodes
   scm-backup-df    :  check freespace on server and all backup nodes

   scm-backup-postfix-start  
    
    
   scm-backup-bootstrap   <no arguments>
          rsync the tarballs from the backup node of the designated server node
          for this node and recover from them


   scm-backup-nightly-as-root :   does it as root ... as done in the crontab
   scm-backup-all-as-root :   does the below as root ... as done in the crontab
    
   scm-backup-all :   invokes the below 
      scm-backup-repo
      scm-backup-trac
      scm-backup-folder   for the apache-confdir   
      scm-backup-purge   : retain the backups from the last 7 days only
      
   scm-recover-all  <fromnode>
   
      NB the folders are not recovered by this, as is installation
      specific, nevertheless tis important that the users file is
      backed up...

      BUT it does invoke scm-recover-users 

   scm-recover-users <fromnode>

      extract the users file from the last svnsetup tarball, 
      called by scm-recover-all
      NB the other svnsetup files are sourced from the repository 
      and contain system specific paths ... so more direct to re-generate 
      them rather than using the backups  

      The users file is different because it is edited thru the webadmin 
      interface     

  
   scm-recover-folders <fromnode>
       still experimental .. NEEDS FURTHER CHECKING PRIOR TO REAL USAGE
       
       recovers the users and permissions files from the last backup
  
   scm-recover-lastlinks  <typ>     
      
      <typ> defaults to tar.gz
      
       this must be run from the backup folder that should contain
       the "last" link eg :
            /var/scm/backup/cms01/tracs/env
                    last -> 2008/08/14/174749
     
       if the "last" link exists then exit without doing anything, 
       however if the last link has been collapsed into a folder 
       (eg by web transfers or non-careful copying) 
       then delete that folder and attempt to recreate the 
       "last" link to the directory containing the last file of type
       
  
  
   scm-backup-rsync :   to the paired node
         to override and send the backup to non-standard destination,
        eg while not inside home internal network need to use G3R
             BACKUP_TAG=G3R scm-backup-rsync


   scm-backup-rsync-from-node : 
                   rsync the backups from a remote node 

   scm-backup-dybsvn-from-node : 
                  copy over the reps for a specific day 


  Common issues ...
  
     1) backups stopped :
    
    compare :
        scm-backup-du
        scm-backup-rls
    check base/cron.bash ... usually some environment change has broken the env setup for cron
    after modifications reset the cron backups..
    
       cron-
       cron-usage
       cron-backup-reset
       cron-list root
       cron-list blyth
   
    2) backups done but not synced off box
     
     
     Probably the agent needs restarting.. this is needs to be done manually after a reboot
     see 
        ssh--usage
        ssh--agent-start
     then check offbox passwordless access with
        scm-backup-
        scm-backup-rls
      
         
  Do an emergency backup and rsync, with :
  
    scm-backup-all-as-root 
    scm-backup-rsync       
    scm-backup-rls      ## check the remote tgz


  TODO : 
  
     1) divided reposnsibilities between here and cron.bash is a mess
     2) not easy to add things to crontab because of this morass 



EOU

}

scm-backup-env(){
   elocal-
   python-
   apache-
}


scm-backup-du(){
    find $SCM_FOLD -name '*.gz' -exec du -h {} \;
}

scm-backup--(){
  sudo bash -lc "export HOME=$HOME ; export ENV_HOME=$(env-home) ; . $ENV_HOME/env.bash ; env- ; scm-backup- ; $*  "
}


scm-backup-all-as-root(){

  sudo bash -lc "export HOME=$HOME ; export ENV_HOME=$(env-home) ; . $ENV_HOME/env.bash ; env- ; scm-backup- ; scm-backup-all  "

}

scm-backup-nightly-as-root(){

  sudo bash -lc "export HOME=$HOME ; export ENV_HOME=$(env-home) ; . $ENV_HOME/env.bash ; env- ; scm-backup- ; scm-backup-nightly  "

}

scm-backup-postfix-start(){

  sudo postfix start 
}


scm-backup-locked-dir(){ echo $SCM_FOLD/LOCKED ; }
scm-backup-is-locked(){
   local msg="=== $FUNCNAME :"
   [ -z "$SCM_FOLD" ] && echo $msg ABORT SCM_FOLD not defined && return 0    ## yep locking is conservative
   if [ -d "$SCM_FOLD/LOCKED" ]; then 
       echo $msg GLOBALLY locked $SCM_FOLD/LOCKED
       ls -alst "$SCM_FOLD/LOCKED" 
       return 0 
   fi
   return 1
}
scm-backup-lock(){
   local caller=$1
   local msg="=== $FUNCNAME :"
   [ -z "$SCM_FOLD" ] && echo $msg ABORT SCM_FOLD not defined && return 1
   local meta=${caller}-started-$(date +"%Y-%m-%d@%H:%M:%S")  
   mkdir -p $SCM_FOLD/LOCKED
   local lock=$SCM_FOLD/LOCKED/$meta
   touch $lock
   echo $msg $lock
}
scm-backup-unlock(){
   local caller=$1
   local msg="=== $FUNCNAME :"
   [ -z "$SCM_FOLD" ] && echo $msg ABORT SCM_FOLD not defined && return 1
   if [ -d "$SCM_FOLD/LOCKED" ]; then 
       echo $msg $caller
       rm -rf "$SCM_FOLD/LOCKED"
   else
       echo $msg $caller WARNING not locked
   fi 
}




scm-backup-all(){
   
   local msg="=== $FUNCNAME :"
   local stamp=$(base-datestamp now %Y/%m/%d/%H%M%S)
   local base=${SCM_BACKUP_TEST_FOLD:-$SCM_FOLD/backup/$LOCAL_NODE}   ## SCM_BACKUP_TEST_FOLD not standardly set, use inline for interactive checking 

   echo $msg starting from pwd $PWD

   scm-backup-is-locked && echo $msg ABORT due to lock && return 1 
   scm-backup-lock $FUNCNAME

   ## remove semaphore is set 
   env-abort-clear
 
   python-
   sqlite-
  
   which python
   echo $LD_LIBRARY_PATH | tr ":" "\n"
 
   local typs="svn repos tracs"
   #local typs="tracs"    ## TEMPORARY CHANGE WHILE DEBUGGING 
   for typ in $typs
   do
       for path in $SCM_FOLD/$typ/*
       do  
           env-abort-active-  && echo $msg ABORTING via file semaphore && return 1  
           if [ -d $path ]; then 

               local name=$(basename $path)
               local inhibiter=$(dirname $path)/${name}-scm-recover-repo

               if [ -L $inhibiter ]; then
                    echo $msg INHIBIT BACKUP of recovered environment, delete the inhibiter $inhibiter to backup this environment $path
               elif [ "$name" == "LOCKED" ]; then
                    echo $msg IGNORE THE LOCKED FOLDER
               elif [ "$LOCAL_NODE" == "cms02" -a "$typ" == "svn" ]; then
                    echo $msg SKIP BACKUP of alien environment $typ at $path on $LOCAL_NODE
               elif [ "$name" == "dybsvn" -a "$NODE_TAG" == "C2R" ]; then 
                    echo $msg skip the slow dybsvn whilst on C2R 
               else
   
                    local starttime=$(scm-backup-date)
                    local progress=$(scm-backup-locked-dir)/$FUNCNAME-$typ-$name-started-$(date +"%Y-%m-%d@%H:%M:%S")  
                    [ ! -d "$(dirname $progress)" ] && echo $msg ABORT no dir for $progress && return 1
                    touch $progress

                    case $typ in 
                         tracs) scm-backup-trac $name $path $base $stamp || return $?  ;;
                     repos|svn) scm-backup-repo $name $path $base $stamp || return $?  ;;
                             *) echo $msg ERROR unhandled typ $typ ;;
                    esac  
                    local endtime=$(scm-backup-date)
                    scm-backup-date-diff "$starttime" "$endtime"
                    
                    progress=$(scm-backup-locked-dir)/$FUNCNAME-$typ-$name-completed-$(date +"%Y-%m-%d@%H:%M:%S")  
                    [ ! -d "$(dirname $progress)" ] && echo $msg ABORT no dir for $progress && return 1
                    touch $progress

  	       fi
           else
  		       echo $msg $typ === skip non-folder $path
  	   fi
       done
   done
   env-abort-active- && echo $msg ABORTING via file semaphore && return 1  
   
   svn-
   
   local dir=$(svn-setupdir)
   local name=$(basename $dir)
   scm-backup-folder $name $dir $base $stamp
   scm-backup-purge $LOCAL_NODE

   scm-backup-unlock $FUNCNAME

}


scm-backup-info(){

   local t=${1:-$NODE_TAG}
   cat << EOI

   local-server-tag   : $(local-server-tag  $t)
        the tag of the node on which the primary server currently resides
        
   local-tag2node \$(local-server-tag)  : $(local-tag2node $(local-server-tag $t)) 

   local-restore-tag  : $(local-restore-tag $t)
        the tag of the node that is housing the backup tarballs from the primary server

   local-tag2node \$(local-restore-tag)  : $(local-tag2node $(local-restore-tag $t)) 



EOI

}

scm-backup-bootstrap(){

   local msg="=== $FUNCNAME :"

   local restore_tag=$(local-restore-tag)
   local server_tag=$(local-server-tag)
   local server_node=$(local-tag2node $server_tag)

   scm-backup-info

   [ -z "$restore_tag" ] && echo $msg ABORT no restore_tag && return 1
   [ -z "$server_tag"  ] && echo $msg ABORT no server_tag && return 1
   [ -z "$server_node" ] && echo $msg ABORT no server_node && return 1

   scm-backup-rsync-from-node $restore_tag "$server_node/"
   scm-recover-all $server_node
}

scm-recover-exclude(){
  case $1 in
    dyw) echo "YES" ;;
      *) echo "NO"  ;; 
  esac
}


scm-recover-all(){

   local msg="=== $FUNCNAME :"
   local fromnode=$1
   [ "$fromnode" == "" ] && echo scm-recover-all needs a fromnode  && return 1   

   local ans
   read -p "$msg using tarballs from node $fromnode ? Enter YES to proceed " ans
   [ "$ans" != "YES" ] && echo $msg ABORTing && return  1

   ## checking trac installation 
   trac-
   trac-check
   trac-inherit-setup

   local types="repos svn tracs"
   for type in $types
   do
      local base
      if [ "$fromnode" == "dayabay"  ]; then
          base=$(scm-backup-tdir)/$fromnode/$type    ## updated manually by dayabay-pull from C2R (from the parasitic C1 backups from IHEP) 
      else
          base=$SCM_FOLD/backup/$fromnode/$type
      fi   

      local lockd=$(dirname $base)/LOCKED
      [ -d "$lockd" ] && echo $msg ABORTING as locked && ls -alst $lockd  && return 1

      local dest=$SCM_FOLD/$type
      local user=$(apache-user)
      
      [ ! -d $dest ] && $SUDO mkdir -p $dest && [ "$SUDO" != "" ] && $SUDO chown $user:$user $dest       
      
      for path in $base/*
      do  
	     if [ -d $path ]; then 
                local name=$(basename $path)
	        if [ "$(scm-recover-exclude $name)" != "YES"  ]; then 
                     scm-recover-repo $name $path $dest
                else
		     echo === scm-recover-all skip excluded, see ... scm-recover-exclude $name : $(scm-recover-exclude $name)
                fi
	     else
		     echo === scm-recover-all skip non-folder $path  
	     fi   
		  #
		  #  eg:
		  #    name : "workflow" 
		  #    path : /var/scm/backup/g4pb/tracs/workflow   or repos equivalent    
          #    dest : /var/scm/tracs or /var/scm/repos 
	  done
      
   done 

   if [ "$fromnode" == "dayabay" ]; then
      echo $msg auto mated usersfile recovery is not possible from $fromnode ... you will need to manually merge usersfiles, see $(env-wikiurl)/ServerHistory 
   else
      scm-recover-users $fromnode
   fi

}



scm-recover-users(){

  local msg="=== $FUNCNAME :"
  local fromnode=${1:-dummy}
  [ "$fromnode" == "dummy" ] && echo scm-recover-users needs a fromnode argument && return 1
  local iwd=$PWD
  local tmp=/tmp/$FUNCNAME && mkdir -p $tmp
  local usr=svnsetup/users.conf
  local tgz=`local-scm-fold`/backup/$fromnode/folders/svnsetup/last/svnsetup.tar.gz
  echo $msg recovering $usr from tgz $tgz into $tmp
  
  cd $tmp
  tar zxvf $tgz $usr 
  cd $iwd

  local dir=`apache-confdir`
  [ ! -d "$dir" ] && echo $msg ERROR no apache-confdir $dir && return 1 

  local cur=$dir/$usr
  local rec=$tmp/$usr
  local ans 
  if [ -f "$cur" ]; then
      diff $cur $rec 
      read -p "Replace existing users file $cur with recovered one $rec , YES to proceed " ans
  else
      read -p "Recover users file $rec , YES to proceed " ans
  fi

  [ "$ans" != "YES" ] && echo $msg SKIPPING && return 1

  

  local cmd="sudo mkdir -p $(dirname $cur) ; sudo cp $rec $cur "
  echo $cmd
  eval $cmd
  apache-chown $cur
}



scm-recover-folders(){
  
   local msg="=== $FUNCNAME :"
   local fromnode=${1:-dummy}
   [ "$fromnode" == "dummy" ] && echo scm-recover-all needs a fromnode argument && return 1
   
   local base=$SCM_FOLD/backup/$fromnode/folders
   for path in $base/*
   do
      if [ -d $path ]; then
         local name=$(basename $path)
         local dest=$(scm-recover-destination $name)
         [ -z $dest ]   && echo $msg ABORT no destination for name $name path $path && return 1
         
             mkdir -p $dest   ## TESTING ONLY
             
         [ ! -d $dest ] && echo $msg ABORT dest $dest does not exist    && return 1  
     
         scm-recover-repo $name $path $dest 
      else
         echo $msg  skip non-folder $path  
      fi
  done

}

scm-recover-destination(){
  case $1 in 
         local|svnsetup|apache2) echo /tmp/$FUNCNAME/$(dirname $(svn-setupdir)) ;;
  esac  
  ## local name still in use on G, apache2 on H,  svnsetup elsewhere 
}


scm-backup-prune(){
  local msg="=== $FUNCNAME :"
  local node=${1:-$LOCAL_NODE} 
  local cmd="sudo find $SCM_FOLD/backup/$node -type d -empty -depth -exec rmdir {} \; "
  echo $msg $cmd
  local ans
  read -p "Enter YES to delete empty dirs with above command :" ans
  [ "$ans" != "YES" ] && echo $msg skipping && return
  echo $msg Proceeding..
  eval $cmd
}


scm-backup-purge(){

  #
  #   deletes backup tgz and containing stamp folders   
  #   such that nmax remain for each repository and tracitory
  #
  #   bash array handling reference : http://tldp.org/LDP/abs/html/arrays.html
  #

  local node=${1:-$LOCAL_NODE} 
  local nmax=4
  local name
  local tgzs
  local itgz
  local ntgz 
  
  ## the bash version on hfag dies, if this is inside the for loop
  declare -a tgzs


  echo ======= scm-backup-purge =====   

  for path in $SCM_FOLD/backup/$node/{tracs,repos,svn,folders}/* 
  do
     [ ! -d "$(dirname $path)" ] && echo skip $path && continue	  
     cd $path 
     
     name=$(basename $path)
     tgzs=($(find . -name '*.tar.gz' | sort ))
     ntgz=${#tgzs[@]}
     
     echo path:$path name:$name ntgz:$ntgz nmax:$nmax
     itgz=0
     while [ "$itgz" -lt "$ntgz" ]
     do    
        local tgz=${tgzs[$itgz]}
        if [ $(( $ntgz - $itgz > $nmax )) == 1 ]; then 
           local container=$(dirname $tgz) 
           local cmd="rm -rf $container"
           echo delete $tgz ... $cmd
           eval $cmd 
        else
           echo retain $tgz
        fi 
        let "itgz = $itgz + 1"
     done
     
  done
}

scm-backup-rls-(){
  local tag=${1:-$BACKUP_TAG}
  local inst=${2:-""}
  local bkpdir=$(local-scm-fold $tag)/backup/$inst
  local day=$(base-datestamp now %Y/%m/%d)
  local smry=" node $tag ($(local-tag2node $tag)) $bkpdir  invoked from $NODE_TAG ($LOCAL_NODE)"
  if [ "$tag" == "IHEP" -o "$tag" == "$NODE_TAG" ] ; then
     echo $msg local $smry
     find $bkpdir -name '*.gz' -exec du -hs {} \; | grep $day
  else
     ssh--
     ! ssh--agent-check && echo $msg ABORT SSH AGENT PROBLEM ... remote $smry && return 1 
     echo $msg remote $smry
     ssh $tag "find  $bkpdir -name '*.gz' -exec du -hs {} \; | grep $day"
  fi
}

scm-backup-rls(){
   local msg="=== $FUNCNAME :"
   local tags=${1:-$BACKUP_TAG}
   local inst=${2:-$(local-node)}
   [ -z "$tags" ] && echo $msg ABORT no backup node has been defined for node $LOCAL_NODE && return 1
   local tmpd=/tmp/$FUNCNAME && mkdir -p $tmpd
   python-
   local tag
   for tag in $tags ; do
       if [ "$tag" == "S" ]; then
          echo $msg exclude sending mail to restricted account/node with tag $tag
       else 
          local tmp=$tmpd/${tag}.txt
          scm-backup-rls- $tag $inst > $tmp 2>&1 
          python-sendmail $tmp

          if [ "$?" != "0" ]; then
             echo $msg FAILED TO SEND NOTIFICATION EMAIL
             if [ "$NODE_TAG" == "G" ]; then
                /usr/local/bin/growlnotify -s -m "$msg FAILED TO SEND NOTIFICATION EMAIL : sudo /usr/sbin/postfix start ?  "
             fi
          fi

          if [ "$NODE_TAG" == "G" ]; then
             cat $tmp | grep workflow | /usr/local/bin/growlnotify -s
          fi

       fi
   done
}


scm-backup-mail(){

  local msg="=== $FUNCNAME :"
  echo $msg DEPRECATED ... USE scm-backup-rls DIRECTLY
  scm-backup-rls 
  
}


scm-backup-parasitic(){

   ##
   ## this is called parasitic because it is used to monitor transfers that i do not have control off ...
   ## ...  i just receive the tarballs
   ##

   local server=$1
   local backup=$2

   local msg="=== $FUNCNAME :"
   local tmp=/tmp/${FUNCNAME}_${server}_${backup}.txt

   echo $msg monitoring to $tmp and sending mail 
   scm-backup-parasitic- $server $backup > $tmp 2>&1
   python-sendmail $tmp
}


scm-backup-parasitic-(){
   local msg="=== $FUNCNAME :"
   local smry="$msg $server $(local-tag2node $server) -> $backup $(local-tag2node $backup) : rsync transfer monitoring   "
   local server="$1"
   local backup="$2"
   local cmd="scm-backup-rls- $backup $(local-tag2node $server)"

   cat << EOP
$smry
Running at : $(date)    
On host    : $(hostname)
Monitoring the rsync transfer from server to backup :
   server : $server $(local-tag2node $server) 
   backup : $backup $(local-tag2node $backup) 
  
Rerun with :
     $cmd
after setting up on C2R ... 
    env-
    scm-backup-
    scm-backup-rls- C dayabay

Most frequent cause of missing tarballs is server outage resulting in 
the ssh-agent from being killed.  
To resolve

  1)  ctrl-cmd-W   SSH into web server 
  2)  uptime       correlate downtime with when tarballs stopped arriving, check cronlogs for the permission denied
  3)  sas          start agent entering passphrase to add identities

 
EOP
   
  eval $cmd

}





scm-backup-dir(){
   echo $(local-scm-fold ${1:-$NODE_TAG})/backup  
}

scm-backup-tdir(){
   ## non-standard to handle travelling other nodes, ie a 2nd transfer of backups not from either of the partaking nodes
   case ${1:-$NODE_TAG} in 
     C2|C2R) echo /data/var/scm/backup ;;
          *) echo /tmp ;;
   esac
}

scm-backup-rsync-all-to-node(){

   local msg="# === $FUNCNAME : "
   local tag=$1
   [ "$tag" == "$NODE_TAG" ] && echo $msg ABORT tag $tag is the same as current NODE_TAG $NODE_TAG ... ABORT && return 1
   
   local cmd="rsync -e ssh -razvt $(scm-backup-dir) $tag:$(scm-backup-dir $tag) "
   echo $cmd

   read -p "$msg Enter YES to proceed " ans
   [ "$ans" != "YES" ] && echo $msg ABORTed && return 1
   echo $msg proceeding ...
   eval $cmd
}



scm-backup-rsync-from-node(){

   local msg="# === $FUNCNAME : "
   local tag=$1
   local node=$2
   ## defaults could be dangerous
   ##local tag=${1:-C}
   ##local node=${2:-dayabay/}
   
   [ "$tag" == "$NODE_TAG" ] && echo $msg ABORT tag $tag is the same as current NODE_TAG $NODE_TAG ... ABORT && return 1
   
   local tgt=$(scm-backup-dir $NODE_TAG)
   mkdir -p $tgt
   local cmd="rsync -e ssh --delete-after -razvt $tag:$(scm-backup-dir $tag)/$node/ $tgt/$node/ "
   echo $cmd

   read -p "$msg Enter YES to proceed " ans
   [ "$ans" != "YES" ] && echo $msg ABORTed && return 1
   echo $msg proveeding ...
   eval $cmd
}

scm-backup-dybsvn-from-node(){

   local msg="# === $FUNCNAME : "
   local tag=${1:-C}
   local dstamp="2008/07/31/122149"
   local stamp=${2:-$dstamp}
   local name="dybsvn"
   local site=$(trac- ; trac-site $name)
   local repos=$(svn-repo-dirname-forsite $site)  
   local orig="hfag"
   
   [ "$tag" == "$NODE_TAG" ] && echo $msg ABORT tag $tag is the same as current NODE_TAG $NODE_TAG ... ABORT && return 1
     
     
   local loc=$(scm-backup-dir $NODE_TAG)  
   local rem=$(scm-backup-dir $tag)
   local reps=$(ssh $tag "ls -1 $rem/$orig/{repos,svn,tracs}/$name/$stamp/$name*.tar.gz ")
   
   echo reps $reps
   
   for rep in $reps
   do
     
      local rel=${rep/$rem\//}
      local tgz=$loc/$rel
      
      echo rep $rep rel $rel tgz $tgz 
      mkdir -p $(dirname $tgz)
      
      local cmd="scp $tag:$rep $tgz"
      echo $cmd
      eval $cmd
   done

   cd $loc/$orig
   for dir in "tracs/$name $repos/$name"
   do 
      ln -sf $stamp last 
   done
   

   
}




scm-backup-nightly(){

   local msg="=== $FUNCNAME :"

    echo
    echo $msg $(date)  @@@ scm-backup-checkscp
    scm-backup-checkscp
 
    echo
    echo $msg $(date)  @@@ scm-backup-all 
    scm-backup-all 

    echo
    echo $msg $(date)  @@@ scm-backup-rsync  ... performing transfers that i control 
    #SCM_BACKUP_RSYNC_OPTS="--exclude=dybsvn-*.tar.gz" scm-backup-rsync  
    scm-backup-rsync  
    
    echo
    echo $msg $(date)  @@@ scm-backup-rls
    scm-backup-rls

    echo
    echo $msg $(date)  @@@ scm-backup-parasitic ... monitoring transfers that i do not control... i just receive the tarballs 
    case $NODE_TAG in 
  C2|C2R) scm-backup-parasitic ZZ C ;;
       *) echo $msg no parasitic monitoring is configured on NODE_TAG $NODE_TAG ;;
    esac

    echo
    echo $msg $(date)  @@@ scm-backup-monitor ... fabric remote tarball checking 
    case $NODE_TAG in
           G) scm-backup-monitor- G ;;
      C2|C2R) scm-backup-monitor- C2 ;;
           *) echo $msg scm-backup-monitor not yet implemented on $NODE_TAG ;;
    esac 



}

scm-backup-rsync-dayabay-pull-from-cms01(){

    local msg="=== $FUNCNAME :"
    local from="C"
    [ "$NODE_TAG" != "C2" ] && echo $msg SHOULD BE RUN FROM C2 NOT $NODE_TAG ABORTING && return 1

    local tag 
    local travelnode="dayabay"
    local source=$(scm-backup-dir $from)/$travelnode

    local ans
    read -p "$msg First check for $source/LOCKED on node $from ? Enter YES to proceed if no locks are in force " ans
    [ "$ans" != "YES" ] && echo $msg ABORTing && return  1

    local cmd="sudo rsync -e ssh --delete-after -razvt $from:$source $(scm-backup-tdir) $(scm-backup-rsync-opts) "
    echo $msg pulling travelnode $travelnode from $from with $cmd
    [ "$from" == "$NODE_TAG" ] && echo $msg ABORT cannot rsync pull from self  && return 1
    echo $msg $cmd
    echo $msg need local pw for sudo then sshkey pw for transfer 
    eval $cmd

    # check locally for locks, just in case 
    local lockd=$(scm-backup-tdir)/$travelnode/LOCKED
    [ -d "$lockd" ] && echo $msg ABORT as LOCKED $lockd && ls -alst $lockd && return 1

}



scm-backup-date(){ date -u +"$(scm-backup-date-fmt)" ; }
scm-backup-date-fmt(){ echo %c ; }
scm-backup-date-conv(){
   case $(uname) in 
     Darwin) date -u -j -f "$(scm-backup-date-fmt)" "$1" +%s  ;; 
      Linux) date -u -d"$1" +%s  ;; 
          *) echo ERROR ;;
   esac
}
scm-backup-date-docs(){ cat << EOD
  Test with::
      start=$(scm-backup-date) ; sleep 10 ; end=$(scm-backup-date) ; scm-backup-date-diff "$start" "$end"
  Note that must quote input params with spaces eg   scm-backup-date-diff "$start" "$end" 
EOD
} 
scm-backup-date-diff(){  
   local msg="=== $FUNCNAME :"
   local a=$(scm-backup-date-conv "$1")
   local b=$(scm-backup-date-conv "$2")
   local s=$(( b - a ))
   local m=$(( $s / 60 ))
   local h=$(( $s / 60 / 60 ))
   local d=$(( $s / 60 / 60 / 24 ))
   echo $msg $1 
   echo $msg $2 
   echo $msg $d days / $h hours / $m minutes / $s seconds
}

scm-backup-dna-(){ $(env-home)/base/digestpath.py $* ; }
scm-backup-dna(){
   #
   #   write a sidecar .dna file for the tgz containing a python dict 
   #   with the digest and size
   #
   local tgz=$1
   scm-backup-dna- $tgz > $tgz.dna
}
scm-backup-dnacheck(){
   #
   #   write .dna to somewhere beneath /tmp (to allow checking with only readonly permissions to the tgz)
   #    and diff with existing .dna
   #
   local tgz=$1
   local tmpd=/tmp/$USER/env/$FUNCNAME/   
   local tmpdna=$tmpd/$tgz.dna
   mkdir -p $(dirname $tmpdna)

   local rc
   if [ -f "$tgz" -a -f "${tgz}.dna" ]; then  
      scm-backup-dna- $tgz > $tmpdna
      diff $tgz.dna $tmpdna
      rc=$?
      if [ "$rc" == "0" ] ; then
          echo $msg OK $tgz  
          rm -f $tmpdna          ## only remove if it passes
      else
          echo $msg FAIL $tgz  
          return $rc
      fi 
   else
      echo $msg cannot perform check as no $tgz.dna 
      return 1
   fi 
   return 0
}

scm-backup-dnachecktgzs(){
   #
   #  checks the dna of tgz found beheath the passed directory 
   #
   local msg=" === $FUNCNAME :"
   local name
   local tgz
   find $1 -name '*.tar.gz.dna' | while read name ; do
       tgz=${name/.dna}
       if [ -f "$tgz" ]; then 
           scm-backup-dnacheck $tgz 
       else
           echo $msg $name $tgz NO SUCH TGZ
       fi 
   done
   return 0
}




## tmpfold semaphoring us no use, as locks must travel with the rsync 
scm-backup-rsync-locked-dir(){ echo $(scm-backup-dir)/$LOCAL_NODE/LOCKED ; }
scm-backup-rsync-is-locked(){
   local msg="=== $FUNCNAME :" 
   local lockd=$(scm-backup-rsync-locked-dir)
   if [ -d "$lockd" ]; then
       echo $msg ERROR IS LOCKED 
       ls -alst $lockd
       return 0
   fi
   return 1
}
scm-backup-rsync-lock(){
   local msg="=== $FUNCNAME :" 
   local label=$1
   local lockd=$(scm-backup-rsync-locked-dir)
   local lock=$lockd/${label}-started-$(date +"%Y-%m-%d@%H:%M:%S")  
   mkdir -p "$lockd" 
   touch $lock
   echo $msg LOCKING $lock
   ls -alst $lockd
}
scm-backup-rsync-unlock(){
   local msg="=== $FUNCNAME :" 
   local lockd=$(scm-backup-rsync-locked-dir)
   echo $msg UNLOCKING $lockd
   ls -alst $lockd
   [ "$(basename $lockd)" != "LOCKED" ] && echo $msg SANITY CHECK FAILS for lockd $lockd && return 1
   rm -rf "$lockd"
}


scm-backup-essh(){
    local tgt=$1
    local port=$(local-port-sshd $tgt)
    [ -z "$port" ] && port=22
    case $port in
      22) echo "-e ssh"  ;;
       *) echo "-e \"ssh -p $port\"" ;;
    esac
}

scm-backup-rsync-opts(){
   case $1 in 
     A|Z9) echo ${SCM_BACKUP_RSYNC_OPTS:-}  --rsync-path=/opt/bin/rsync ;; 
        *) echo ${SCM_BACKUP_RSYNC_OPTS:-}  ;;
   esac	   
}

scm-backup-rsync(){

   # 
   # rsync the local backup repository to an off box mirror on the paired $BACKUP_TAG node 
   #   - have to set up ssh keys to allow non-interactive sshing 
   # 

   local msg="=== $FUNCNAME :" 
   scm-backup-is-locked && echo $msg ABORT due to lock && return 1 
   scm-backup-lock $FUNCNAME

   ssh--
   ! ssh--agent-check && echo $msg ABORT ssh--agent-check FAILED : seems that you are not hooked up to your ssh-agent : possible NODE mischaracterization &&  ssh--envdump && return 1

   local tags=${1:-$BACKUP_TAG}   
   [ -z "$tags" ] && echo $msg ABORT no backup node\(s\) for NODE_TAG $NODE_TAG see base/local.bash::local-backup-tag && return 1

   local tag 
   for tag in $tags ; do
 
       [ "$tag" == "$NODE_TAG" ] && echo $msg ABORT cannot rsync to self  && return 1

       ## NB the rsync lock is distinct from the above global lock 
       echo 
       scm-backup-rsync-is-locked && echo $msg scm-backup-rsync ABORTING as locked  && return 1 
       scm-backup-rsync-lock ${FUNCNAME}-starting-to-$tag

       local remote=$(scm-backup-dir $tag)
       local source=$(scm-backup-dir)/$LOCAL_NODE

       ## have to skip from XX as do not have permission to ssh 
       [ $NODE_TAG != "XX" ] && ssh $tag "mkdir -p  $remote"

       local starttime=$(scm-backup-date)
       echo $msg starting transfer to tag $tag at $starttime
       echo $msg transfer $source to $tag:$remote/ 

       local essh=$(scm-backup-essh $tag)
       local opts=$(scm-backup-rsync-opts $tag)
       local cmd="time rsync $essh --delete-after --stats -razvt $source $tag:$remote/ $opts  "
       echo $msg $cmd
       eval $cmd

       scm-backup-rsync-unlock ${FUNCNAME}-finished-to-$tag

       echo $msg quick re-transfer $source to $tag:$remote/ after unlock ## shoud be very quick as should be just removing the remote LOCKED dir
       echo $msg $cmd
       eval $cmd
 
       if [ "$NODE_TAG" == "XX"  ]; then 
           echo $msg skip remote DNA check as lack ssh permissions
       else
           case $tag in 
               H1)  echo $msg skip invoke remote DNA check to destination $tag  ;;
                S)  echo $msg skip invoke remote DNA check to destination $tag  ;;
               G3)  echo $msg remote DNA check && ssh $tag "export ENV_HOME=~/env ; . ~/env/env.bash && env-env && hostname && uname && date && scm-backup- && scm-backup-dnachecktgzs $remote/$LOCAL_NODE "   ;;
                *)  echo $msg remote DNA check && ssh $tag ". ~/env/env.bash && env-env && hostname && uname && date && scm-backup- && scm-backup-dnachecktgzs $remote/$LOCAL_NODE "   ;;
           esac
       fi 

       local endtime=$(scm-backup-date)
       scm-backup-date-diff "$starttime" "$endtime"
  done 

  scm-backup-unlock $FUNCNAME
}


scm-backup-checkssh(){

   local tags=${1:-$BACKUP_TAG}   
   [ -z "$tags" ] && echo $msg ABORT no backup node\(s\) for NODE_TAG $NODE_TAG see base/local.bash::local-backup-tag && return 1
 
   local tag 
   for tag in $tags ; do
       [ "$tag" == "$NODE_TAG" ] && echo $msg ABORT cannot rsync to self  && return 1
       local remote=$(scm-backup-dir $tag)
       local cmd="ssh $tag df -h $remote" 
       echo;echo $msg $cmd
       eval $cmd
  done 


}

scm-backup-checkscp(){

   local tags=${1:-$BACKUP_TAG}   
   [ -z "$tags" ] && echo $msg ABORT no backup node\(s\) for NODE_TAG $NODE_TAG see base/local.bash::local-backup-tag && return 1
 
   local tag 
   local nonce=/tmp/env/$FUNCNAME/$FUNCNAME.$(hostname).txt
   mkdir -p $(dirname $nonce) 
   touch $nonce

   for tag in $tags ; do
       [ "$tag" == "$NODE_TAG" ] && echo $msg ABORT cannot rsync to self  && return 1
       local cmd="scp $nonce $tag:/tmp/" 
       echo;echo $msg $cmd
       eval $cmd
  done 


}



scm-backup-rsync-xinchun(){

   # 
   # rsync the local backup repository to an off box mirror on the paired $BACKUP_TAG node 
   #   - have to set up ssh keys to allow non-interactive sshing 
   # 
   #  hmm the passwordless ssh is not setup for "root" user , so have to do this as me, but the above backup as root
   #

   local msg="=== $FUNCNAME :" 
   local tag=${1:-$BACKUP_TAG}   
  
   [ -z $tag ] && echo $msg ABORT no backup node for NODE_TAG $NODE_TAG see base/local.bash::local-backup-tag && return 1
   [ "$tag" == "$NODE_TAG" ] && echo $msg ABORT cannot rsync to self  && return 1
  
   local remote=cms01.phys.ntu.edu.tw:/var/scm/backup/ 
   local source=$(scm-backup-dir)/$LOCAL_NODE
 
   if [	"$tag" == "IHEP" ] ; then
     echo $msg transfer $source to $remote/
     local cmd="scp -r $source dayabayscp@$remote/ "    
     eval $cmd
   else
     ssh $tag "mkdir -p  $remote"
     echo $msg transfer $source to $tag:$remote/ 
     local cmd="rsync -e ssh --delete-after -razvt $source $tag:$remote/ "
     echo $msg $cmd
     eval $cmd
  fi
}


scm-backup-sudouser(){

   local msg="=== $FUNCNAME :"
   
   local user=$(apache-user)
   [ -z $user ] && echo $msg ERROR apache-user not defined && return 1 

   local sudouser
   if [ "$SUDO" == "" ]; then
      sudouser=""
    else
      sudouser="$SUDO -u $user"
    fi
    echo $sudouser
}



scm-recover-lastlinks(){

   local msg="=== $FUNCNAME :"
   local typ=${1:-tar.gz}
   
   [ -L last ] && echo $msg last links in path $path already present ... nothing to do && return 0
   [ -d last -a ! -L last ] && echo $msg deleting directory && rm -rf last

   local lst=$(scm-backup-last-of-type $typ)
   local dst=$(dirname $lst)

   [   -z $dst ] && echo $msg ERROR no last $typ found     && return 1
   [ ! -d $dst ] && echo $msg ERROR no such directory $dst && return 2

   echo $msg planting last link to dst $dst in $PWD 
   ln -sf $dst last 
   
}


scm-backup-last-of-type(){
   
   local typ=${1:-tar.gz}
   declare -a list
   list=($(find . -name "*.$typ" | sort)) 
   local n=${#list[@]}
   local m=$(($n - 1))
      
   if [ $m -gt -1 ]; then
      local last=${list[$m]}
      echo $last
   else
      echo -n
   fi
}


scm-recover-repo(){

   local msg="=== $FUNCNAME :"
   local name=${1:-dummy}   ## name of the backup
   local path=${2:-dummy}   ## absolute path to backup folder containing the last link  
   local dest=${3:-dummy}   ## destination folder, usually $SCM_FOLD/repos OR $SCM_FOLD/tracs 
   local type=$(basename $dest)
   
   [ "$name" == "dummy" ] && echo $msg ERROR the name must be given && return 1 
   [ ! -d "$path" ]       && echo $msg ERROR path $path does not exist && return 1 
   [ ! -d "$dest" ]       && echo $msg ERROR destination folder $dest does not exist && return 1 
   
   local sudouser=$(scm-backup-sudouser)
   
   cd $path
   
   # recover collapsed links
   scm-recover-lastlinks tar.gz
   
   local stamp=$(readlink last)
   local target_fold=$path/$stamp
   cd $target_fold
   
   if [ "$?" == "1" ]; then
      echo $msg error target_fold $target_fold not found 
   else
   
      declare -a tgzs
      tgzs=($(ls -1 *.tar.gz))
      local ntgz=${#tgzs[@]}

      if [ "$ntgz" == "1" ]; then
      
         local tgz=${tgzs[0]} 
         local tgzname=${tgz%.tar.gz}
         local tgzpath=$target_fold/$tgzname.tar.gz
      
         cd $dest 
         
         if [ -d "$name" ]; then
            echo $msg the repository/folder:$name is present already in dest $dest , must delete the $name folder before can recover 
            echo $msg stamp $stamp target_fold $target_fold ==== tgz $tgz ===== tgzname $tgzname
         else
             
            echo $msg recovering repository/folder $name from tarball $tgzpath $tgzname into $(pwd) sudouser:[$sudouser] SUDO:[$SUDO]
            $SUDO cp $tgzpath .
            $SUDO tar zxf $tgzname.tar.gz
            
            ## document the recovery via a link to the backup tarball
            $SUDO ln -sf $tgzpath ${name}-scm-recover-repo
            
            $SUDO rm -f $tgzname.tar.gz
        
            ## svn tarballs have the revision number appended to their names
            if [ "$tgzname" != "$name" ]; then
              $SUDO mv $tgzname $name
            fi

            apache-chown $name -R
            #local user=$(apache-user)
            #$SUDO chown -R $user $name
          
            if [ "$type" == "tracs" ]; then
                 scm-backup-synctrac $name
             fi
         fi      
      else
         echo $msg  ERROR there is not 1 tgz in target_fold $target_fold
      fi 
   fi
}



scm-backup-uninhibit-(){
   cat << EOC
rm  $SCM_FOLD/tracs/*-scm-recover-repo
rm  $SCM_FOLD/svn/*-scm-recover-repo
rm  $SCM_FOLD/repos/*-scm-recover-repo
EOC

}

scm-backup-uninhibit(){
    local msg="=== $FUNCNAME :"
    echo $msg SCM_FOLD $SCM_FOLD NODE_TAG $NODE_TAG

    ls -alst $SCM_FOLD/{tracs,repos,svn}/*-scm-recover-repo
    echo $msg proceed with removal of inhibitors ....
    scm-backup-uninhibit-
    local ans

    read -p "$msg enter YES to proceed " ans
    [ "$ans" != "YES" ] && echo $msg skipping && return 0

    local cmd
    scm-backup-uninhibit-  | while read cmd ; do 
       echo $cmd
       eval $cmd
    done 


}


scm-backup-synctrac(){
  
     local msg="=== $FUNCNAME :"
     local name=${1:dummy}
     [ "$name" == "dummy" ] && echo $msg ERROR the name must be given && return 1 
     echo $msg invoking trac-configure-instance for $name to customize server specific paths etc..
     trac-

     ## setting permissions 
     $SUDO find $(trac-envpath $name) -type d -exec chmod go+rx {} \;
     SUDO=$SUDO trac-configure-instance $name
               
     echo $msg resyncing the instance with the repository ... as repository_dir has changed ... avoiding the yellow banner
     TRAC_INSTANCE=$name trac-admin-- resync

     echo $msg ensure everything in the envpath is accessible to apache ... resyncing sets ownership of trac.log to root 
     apache-
     sudo find $(trac-envpath $name) -group root -exec chown $(apache-user):$(apache-group) {} \; 

}



scm-backup-repo(){

   local iwd=$PWD
   local msg="=== $FUNCNAME :" 
   local name=${1:-dummy}   ## name of the repo
   local path=${2:-dummy}   ## absolute path to the repo  
   local base=${3:-dummy}   ## backup folder
   local stamp=${4:-dummy}  ## date stamp
   local site=$(trac- ; trac-site $name)
   
   echo $msg name $name path $path base $base stamp $stamp site $site ===
   
   [ "$name" == "dummy" ]  &&  echo $msg ERROR the name must be given && return 1 
   [ ! -d "$path" ]        &&  echo $msg ERROR path $path does not exist && return 1 
   [ "$base" == "dummy" ]  &&  echo $msg ERROR the base must be given && return 1 
   [ "$stamp" == "dummy" ] &&  echo $msg ERROR the stamp must be given && return 1 
   
   local target_fold=$base/$(svn-repo-dirname-forsite $site)/$name/$stamp
   #   
   #  
   # hot-copy.py creates tgzs like : 
   #       name-rev.tar.gz 
   #       name-rev-index.tar.gz       index:1,2,3,...
   # 
   #  inside $target_fold , which must exist
   # 
     
   local hot_backup=$(svn-hotbackuppath)      
   [ ! -x $hot_backup ] && echo $msg ABORT no hot_backup script $hot_backup && return 1
                  			  	  
   local rc
   local cmd="mkdir -p $target_fold &&  $hot_backup --archive-type=gz $path $target_fold && cd $base/$(svn-repo-dirname-forsite $site)/$name && rm -f last && ln -s $stamp last "   
   echo $msg $cmd
   eval $cmd
   rc=$? 

   [ "$rc" != "0" ] && echo $msg ERROR $rc && return $rc

   local rev=$(svnlook youngest $path)
   local tgz=${target_fold}/${name}-${rev}.tar.gz
   scm-tgzcheck-ztvf $tgz
   rc=$?
   [ "$rc" != "0" ] && echo $msg tgz $tgz rev $rev integrity check failure $rc && return $rc 
   echo $msg tgz $tgz rev $rev integrity check ok 
   
   scm-backup-dna $tgz

   cd $iwd
   return 0
}


scm-backup-trac(){

   local iwd=$PWD
   local msg="=== $FUNCNAME :" 
   local name=${1:-dummy}     ## name of the trac
   local path=${2:-dummy}     ## absolute path to the trac
   local base=${3:-dummy}     ## backup folder
   local stamp=${4:-dummy}  ## date stamp
   
   echo $msg name $name path $path base $base stamp $stamp === $(date)
   
   #
   #  perhaps the stamp should be above the name, and have only one stamp 
   #
   
   [ "$name" == "dummy" ] &&  echo the name must be given && return 1 
   [ ! -d "$path" ]       &&  echo ERROR path $path does not exist && return 1 
   [ "$base" == "dummy" ] &&  echo the base must be given && return 1 
   [ "$stamp" == "dummy" ] &&  echo the stamp must be given && return 1 
   
   
   local source_fold=$path
   local target_fold=$base/tracs/$name/$stamp/$name
   local parent_fold=$(dirname $target_fold)

   trac-
   trac-admin-sqlite-check || env-abort 
   
   ## target_fold must NOT exist , but its parent should
   ## too many pythons around to rely on an external PYTHON_HOME
   
   local tracadmin=$(which trac-admin)
   local pymode=$(python-mode)
   if [ "$pymode" == "source" ]; then
     [ "$(python-home)/bin/trac-admin" != "$tracadmin" ] && echo $msg ERROR wrong source tracadmin $tracadmin ... env screwup  && return 1  
   elif [  "${pymode:0:6}" == "system" ] ; then
     case $tracadmin in
              /usr/bin/trac-admin) echo -n ;;
        /usr/local/bin/trac-admin) echo -n ;;
                                *) echo $msg ERROR wrong system trac_admin $tracadmin ... env screwup  && return 1
     esac 
   fi
   
   [ ! -x $tracadmin ] && echo $msg ABORT no trac_admin at $tracadmin && return 1
 
   
   local rc
   ## curious : sometimes (depends on cwd i assume)  the rampant wildcard "tar -zcf $name.tar.gz $name/*" causes an error ... changing to "tar -zcf $name.tar.gz $name" results in same structure anyhow (maybe dotfile diff ?)
   local cmd="mkdir -p $parent_fold && $tracadmin $source_fold hotcopy $target_fold && cd $parent_fold && tar -zcf $name.tar.gz $name && rm -rf $name && cd $base/tracs/$name && rm -f last && ln -s $stamp last "
   echo $msg $cmd
   eval $cmd 
   rc=$?
   [ "$rc" != "0" ] && echo $msg trac hotcopy failure $rc && return $rc 

   local tgz=${parent_fold}/${name}.tar.gz
   scm-tgzcheck-trac ${name} ${tgz}
   rc=$?
   [ "$rc" != "0" ] && echo $msg trac tgzcheck failure $rc && return $rc 

   scm-backup-dna $tgz

   cd $iwd
   return 0   
}

scm-tgzcheck-ztvf(){

   local msg="=== $FUNCNAME :"
   local tgz=$1    ## absolute path to tgz
   local rc
   tar ztvf $tgz > /dev/null 
   rc=$?
   [ "$rc" != "0" ] && echo $msg tgz $tgz integrity check FAILURE $rc && return $rc
   echo $msg OK tgz $tgz integrity check succeeds
   return 0   
}

scm-tgzcheck-trac(){

   local iwd=$PWD
   local msg="=== $FUNCNAME :"
   local name=$1   ## name of trac repo eg env, newtest
   local tgz=$2    ## absolute path to tgz
   local clean=1
   local rc
   local tmp_fold=$(dirname $tgz)/tmp      ## in same dir as tgz

   tar ztvf $tgz > /dev/null 
   rc=$?
   [ "$rc" != "0" ] && echo $msg tgz $tgz integrity check FAILURE $rc && return $rc
   echo $msg OK tgz $tgz integrity check succeeds

   local chk="rm -rf $tmp_fold && mkdir -p $tmp_fold && cd $tmp_fold &&  tar zxf $tgz ${name}/db/trac.db "
   eval $chk
   rc=$?
   [ "$rc" != "0" ] && echo $msg tmp trac.db extraction FAILURE && return $rc
   echo $msg OK tgz $tgz tracdb extraction succeeds

   trac-
   trac-admin-sqlite-check || env-abort     ## LD_LIBRARY_PATH must include dirs for appropriate version of sqlite
 
   local tracdb="${tmp_fold}/${name}/db/trac.db"
   local dumpdb="${tmp_fold}/${name}.sql"
   local dump="echo .dump | sqlite3 ${tracdb} > ${dumpdb}"
   eval $dump
   rc=$?
   [ "$rc" != "0" ] && echo $msg sqlite3 dumping of $tracdb FAILURE && return $rc
   echo $msg OK sqlite3 dumping of $tracdb to ${dumpdb} succeeds

   ## avoid leaving around large dumpfiles 
   if [ "$clean" == "1" ]; then
        [ "$(basename $tmp_fold)" == "tmp" ] && rm -rf $tmp_fold && echo $msg OK removed $tmp_fold modify clean to retain
   fi 

   #
   #  to check integrity of the sqlite database that is the heart of trac
   #   sqlite3 /path/to/env/db/trac.db
   #    > .help
   #    > .tables
   #    > .schema wiki
   #    > .dump            dumps the database as SQL statements 
   # 
   cd $iwd           ## hanging around in dirs that get deleted cause hotcopy failure 
   return 0   
  
}


scm-backup-folder(){

   local msg="=== $FUNCNAME :"
      
   local name=${1:-dummy}     ## 
   local path=${2:-dummy}     ## absolute path of folder to be backed up 
   local base=${3:-dummy}     ## backup folder
   local stamp=${4:-dummy}    ## date stamp
   
   echo $msg name $name path $path base $base stamp $stamp ===
   
   [ "$name" == "dummy" ] &&  echo the name must be given && return 1 
   [ ! -d "$path" ]       &&  echo ERROR path $path does not exist && return 1 
   [ "$base" == "dummy" ] &&  echo the base must be given && return 1 
   [ "$stamp" == "dummy" ] &&  echo the stamp must be given && return 1 
   
   local source_fold=$path
   local target_fold=$base/folders/$name/$stamp
   
   local cmd="mkdir -p $target_fold ; cd $(dirname $source_fold) ; rm -f $name.tar.gz ; tar -zcvf $name.tar.gz $(basename $source_fold)  ; cp $name.tar.gz $target_fold/ && cd $base/folders/$name && rm -f last && ln -s $stamp last "

   echo $msg 
   echo $msg "$cmd"
   eval $cmd

   local tgz=$target_fold/$name.tar.gz
   scm-backup-dna $tgz
 
}


scm-backup-check(){
   local msg="=== $FUNCNAME :"
   local sv=${1:-$(local-server-tag)}
   local bks=$(local-backup-tag $sv)
   local bk ; for bk in $bks ; do
      echo;echo $msg  backups from $sv on $bk
      ssh $bk "find $(scm-backup-dir $bk)/$(local-tag2node $sv) -name '*.tar.gz' -exec du -hs {} \; "
   done
}


scm-backup-df(){
   local msg="=== $FUNCNAME :"
   local sv=${1:-$(local-server-tag)}
   local bks=$(local-backup-tag $sv)
   echo;echo $msg on the server $sv local-var-base : $(local-var-base $sv)
   ssh $sv "df -h "
   
   local bk ; for bk in $bks ; do
      echo;echo $msg  $sv -\> $bk   local-var-base : $(local-var-base $bk)
      ssh $bk "df -h "
   done

}

scm-backup-monitor-(){
   local msg="=== $FUNCNAME :"
   local hub=${1:-C2}
   shift 
   local iwd=$PWD
   cd $(env-home)/scm 
   # the role specifies the hub nodes, such as C2 from where the tarballs emanate  
   local cmd="fab -R $hub scm_backup_monitor "
   echo $msg $cmd
   eval $cmd 
   cd $iwd
}

scm-backup-monitor(){  scm-backup-monitor- C2 ; }
scm-backup-monitorw(){ scm-backup-monitor- G ; }



