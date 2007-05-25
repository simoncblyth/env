
crontab-file(){


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

50 22 25 5 * /sbin/service apache2 configtest
50 22 25 5 * /sbin/service apache  configtest

EOF

#sudo crontab -u root  

sudo crontab -u root -l  > $
sudo crontab -u root $tmp



}
