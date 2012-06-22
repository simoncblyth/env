ntp-usage(){ cat << EOU

EOU
}


ntp-setup(){

   cat << EOH
enter server lines like this , of a stratum 2 ntp server :
# http://ntp.isc.org/bin/view/Servers/StratumTwoTimeServers
server ntp.ucsd.edu

logfile /var/log/ntpd.log

then restart the ntpd service

EOH

  echo sudo vi /etc/ntp.conf
  echo sudo /sbin/service ntpd restart 

  sudo /usr/sbin/ntpd -q -g 

}


ntp-log(){
   cat /var/log/ntpd.log
}


# locate ntpd reveals :
#
#  /usr/sbin/ntpdate
#  /usr/sbin/
#
#

#
#[blyth@hfag env]$  sudo /sbin/service ntpd restart
#ntpd: Removing firewall opening for 127.127.1.0 port 123   [  OK  ]
#ntpd: Removing firewall opening for ntp.ucsd.edu port 123iptables: Bad rule (does a matching rule exist in that chain?)
#                                                           [FAILED]
#Shutting down ntpd:                                        [  OK  ]
#ntpd: Opening firewall for input from 127.127.1.0 port 123 [  OK  ]
#ntpd: Opening firewall for input from ntp.ucsd.edu port 123[  OK  ]
#Starting ntpd:                                             [  OK  ]
#


