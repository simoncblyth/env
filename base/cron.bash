

cron-delete(){
   local user=${1:-root}
   cron-list $user
   sudo crontab -u $user -r -i 
}

cron-list(){
   local user=${1:-root}
   date
   sudo crontab -u $user -l
}

cron-log(){
   sudo cat /var/log/cron
}



cron-setup-backup(){

      local user=${1:-root}

      local crondir=/usr/local/cron/$user
      [ -d $crondir ] || sudo mkdir -p $crondir
      local cronlog=$crondir/$$.log
      local     tmp=$crondir/$$crontab  
  
  
      ## hfag is 20min before the real time 
         
      local       minute=10   # (0 - 59)
      local         hour=18   # (0 - 23)
      local day_of_month="*"  # (1 - 31)
      local        month="*"  # (1 - 12)
      local  day_of_week="*"  # (0 - 7) (Sunday=0 or 7)
      
      local     cmd
      local   delta  

      if [ "$user" == "root" ]; then
         
         cmd="(. $ENV_BASE/$ENV_BASE.bash ; env ; type scm-backup-all ; scm-backup-all     ) > $cronlog 2>&1"
         delta=0
      
      elif [ "$user" == "blyth" ]; then
         
         cmd="(. $ENV_BASE/$ENV_BASE.bash ; env ; type scm-backup-rsync ; scm-backup-rsync ) > $cronlog 2>&1"
         delta=30   
         
      else
         echo cron-setup-backup bad user $user && return 1 
      fi 


      cat << EOT > $tmp
#
SHELL=/bin/bash
PATH=/sbin:/bin:/usr/sbin:/usr/bin
MAILTO=blyth@hep1.phys.ntu.edu.tw
HOME=$HOME
#
$(( $minute + $delta )) $hour $day_of_month $month $day_of_week $cmd
#
EOT
 
      local reply=$(sudo crontab -u $user -l 2>&1)      ## redirection sending stderr onto stdout
      
      if ([ "$reply" == "no crontab for $user" ] || [ "$reply" == "crontab: no crontab for $user" ])  then
           echo =========== initializing crontab for user $user to $tmp 
           cat $tmp 
           sudo crontab -u $user $tmp && sudo cp -f $tmp $crondir/crontab
      else
           echo cannot proceed as a crontab for user $user exists already, must "cron-delete $user" first 
           cron-list $user
      fi

}










cron-test(){

    #
    # defaults to three minutes from now
    # note limitation : assumes not about to go into another hr, day, month etc..
    #
    #  Observations:
    #  1) seems must export variables for them to be visible on the above cron cmdline 
    #  2) the sudo environment is a little funny ... hence this test
    #

    local user=${1:-root}
    shift

    local       def_minute=$(( $(date +"%M") + 3 ))   
    local         def_hour=$(date +"%H")
    local def_day_of_month=$(date +"%d")
    local        def_month=$(date +"%m")

    local       minute=${1:-$def_minute}
    local         hour=${2:-$def_hour}
    local day_of_month=${3:-$def_day_of_month}
    local        month=${4:-$def_month}
    local  day_of_week="*"

    local cronlog=/tmp/crontest
    local  tmp=/tmp/$$crontab

    local cmd 
   
     if [ "$user" == "root" ]; then
         
         cmd="(. $ENV_BASE/$ENV_BASE.bash ; env ; type scm-backup-all ; scm-backup-all     ) > $cronlog 2>&1"
              
     elif [ "$user" == "blyth" ]; then
         
         cmd="(. $ENV_BASE/$ENV_BASE.bash ; env ; type scm-backup-rsync ; scm-backup-rsync ) > $cronlog 2>&1"
      
     else
         echo user $user not handled  && return 1                    
     fi
   
     cat << EOF > $tmp
#
SHELL=/bin/bash
PATH=/sbin:/bin:/usr/sbin:/usr/bin
MAILTO=blyth@hep1.phys.ntu.edu.tw
HOME=$HOME
#
# +---------------- minute (0 - 59)
# |  +------------- hour (0 - 23)
# |  |  +---------- day of month (1 - 31)
# |  |  |  +------- month (1 - 12)
# |  |  |  |  +---- day of week (0 - 7) (Sunday=0 or 7)
# |  |  |  |  |
# *  *  *  *  *  command to be executed
#
$minute $hour $day_of_month $month $day_of_week $cmd 
#
EOF


     local reply=$(sudo crontab -u $user -l 2>&1)      ## redirection sending stderr onto stdout
     if ([ "$reply" == "no crontab for $user" ] || [ "$reply" == "crontab: no crontab for $user" ])  then
          echo =========== initializing crontab for $user to $tmp 
          cat $tmp 
          sudo crontab -u $user $tmp
     else
          echo cannot proceed as a crontab for user $user exists already, do  cron-delete $user / cron-list $user  first 
     fi

}









cron-setup-shutdown(){

      crondir=/usr/local/cron
      [ -d $crondir ] || sudo mkdir -p $crondir
  
      ## hfag is 20min before the real time ... so switch off one hr before 
      ## scheduled off  
   
      local       minute=30  # (0 - 59)
      local         hour=7   # (0 - 23)
      local day_of_month=25  # (1 - 31)
      local        month=5   # (1 - 12)
      local  day_of_week="*" # (0 - 7) (Sunday=0 or 7)

      cronlog=$crondir/$$.log
      tmp=/tmp/$$crontab 

      cat << EOT > $tmp
#
SHELL=/bin/bash
PATH=/sbin:/bin:/usr/sbin:/usr/bin
MAILTO=blyth@hep1.phys.ntu.edu.tw
HOME=/tmp
#
$(( $minute + 0 )) $hour $day_of_month $month $day_of_week /sbin/service apache2 stop  >  $cronlog 2>&1
$(( $minute + 1 )) $hour $day_of_month $month $day_of_week /sbin/service apache  stop >>  $cronlog 2>&1
$(( $minute + 2 )) $hour $day_of_month $month $day_of_week /sbin/service exist   stop >>  $cronlog 2>&1
$(( $minute + 3 )) $hour $day_of_month $month $day_of_week /sbin/service tomcat  stop >>  $cronlog 2>&1
$(( $minute + 4 )) $hour $day_of_month $month $day_of_week  ps -ef                    >>  $cronlog 2>&1
$(( $minute + 5 )) $hour $day_of_month $month $day_of_week /sbin/shutdown -t 10 now   >>  $cronlog 2>&1
#
EOT
 
reply=$(sudo crontab -u root -l 2>&1)      ## redirection sending stderr onto stdout
if ([ "$reply" == "no crontab for root" ] || [ "$reply" == "crontab: no crontab for root" ])  then
   echo =========== initializing crontab for root to $tmp 
   cat $tmp 
   sudo crontab -u root $tmp && sudo cp -f $tmp $crondir/crontab
   
else
   echo cannot proceed as a crontab for root exists already, must "cron-delete" first 
   cron-list
fi

}








