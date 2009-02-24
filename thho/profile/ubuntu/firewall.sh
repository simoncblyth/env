#!/bin/bash
# Simple fire wall used on a personal computer.
# It's not suitable to be used on a server.
#
# usage:
#
#   ln -s firewall.sh /etc/init.d/firewall.sh
#   chmod 775 firewall.sh
#   update-rc.d -f firewall.sh defaults
#
#
#   ref:
#       http://www.ubuntu-tw.org/modules/newbb/viewtopic.php?topic_id=10836
#
#
#PATH=/sbin:/bin:/usr/sbin:/usr/bin
#export PATH
iptables -F
iptables -X
iptables -Z
iptables -P   INPUT DROP
iptables -P  OUTPUT ACCEPT
iptables -P FORWARD ACCEPT
iptables -A INPUT -i lo -j ACCEPT
iptables -A INPUT -m state --state RELATED,ESTABLISHED -j ACCEPT

#iptables -L >> $HOME/iptables_firewall.log
