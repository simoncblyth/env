
crontab-setup(){


if [ "$NODE_TAG" == "G" ]; then
   cmd="$(which apachectl) configtest"
else
   cmd="/sbin/service apache2 configtest"
fi


tmp=/tmp/$$crontab
cat << EOF > $tmp

SHELL=/bin/bash
PATH=/sbin:/bin:/usr/sbin:/usr/bin
MAILTO=blyth@hep1.phys.ntu.edu.tw
HOME=

# Use the hash sign to prefix a comment
# +---------------- minute (0 - 59)
# |  +------------- hour (0 - 23)
# |  |  +---------- day of month (1 - 31)
# |  |  |  +------- month (1 - 12)
# |  |  |  |  +---- day of week (0 - 7) (Sunday=0 or 7)
# |  |  |  |  |
# *  *  *  *  *  command to be executed

50 22 25 5 * $cmd

EOF


reply=$(sudo crontab -u root -l 2>&1)      ## redirection sending stderr onto stdout
if ([ "$reply" == "no crontab for root" ] || [ "$reply" == "crontab: no crontab for root" ])  then
   sudo crontab -u root $tmp
else
   echo cannot proceed as a crontab for root exists already
fi




}
