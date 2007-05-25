

cron-delete(){
   crontab-list
   sudo crontab -u root -r -i 
}

cron-list(){
   date
   sudo crontab -u root -l
}

cron-log(){
   sudo cat /var/log/cron
}

cron-setup(){

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

   cat << EOT > /usr/local/cron/crontab
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
$(( $minute + 4 )) $hour $day_of_month $month $day_of_week /sbin/shutdown -t 10       >>  $cronlog 2>&1

EOT
 

}

cron-test(){

## defaults to three minutes from now
## note limitation : assumes not about to go into another hr, day, month etc..

local       def_minute=$(( $(date +"%M") + 3 ))   
local         def_hour=$(date +"%H")
local def_day_of_month=$(date +"%d")
local        def_month=$(date +"%m")

local       minute=${1:-$def_minute}
local         hour=${2:-$def_hour}
local day_of_month=${3:-$def_day_of_month}
local        month=${4:-$def_month}
local  day_of_week="*"


if [ "$NODE_TAG" == "G" ]; then
   cmd="$(which apachectl) configtest > /tmp/crontest 2>&1"
else
   cmd="/sbin/service apache2 configtest > /tmp/crontest 2>&1"
fi


tmp=/tmp/$$crontab
cat << EOF > $tmp
#
SHELL=/bin/bash
PATH=/sbin:/bin:/usr/sbin:/usr/bin
MAILTO=blyth@hep1.phys.ntu.edu.tw
HOME=/tmp
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


reply=$(sudo crontab -u root -l 2>&1)      ## redirection sending stderr onto stdout
if ([ "$reply" == "no crontab for root" ] || [ "$reply" == "crontab: no crontab for root" ])  then
   echo =========== initializing crontab for root to $tmp 
   cat $tmp 
   sudo crontab -u root $tmp
else
   echo cannot proceed as a crontab for root exists already, do  crontab-delete / crontab-info  first 
fi

}


crontab-setup(){


}


