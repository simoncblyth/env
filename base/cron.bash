

crontab-delete(){
   sudo crontab -u root -r -i 
}

crontab-setup(){

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
   cmd="$(which apachectl) configtest"
else
   cmd="/sbin/service apache2 configtest"
fi


tmp=/tmp/$$crontab
cat << EOF > $tmp
#
SHELL=/bin/bash
PATH=/sbin:/bin:/usr/sbin:/usr/bin
MAILTO=blyth@hep1.phys.ntu.edu.tw
HOME=/tmp
#
# Use the hash sign to prefix a comment
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
   echo cannot proceed as a crontab for root exists already
fi




}
