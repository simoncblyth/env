#!/bin/bash
# Simple fire wall used on a personal computer.
# It's not suitable to be used on a server.
#
# usage:
#   add the line:
#
#   source $HOME/env/thho/profile/ubuntu/firewall.sh
#
#   in /etc/rc.local
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

iptables -L >> $HOME/iptables_firewall.log
