
#
# service-setup     ... do the chkconfig script annotation
# service-list      ... list the chkconfig status
# service-act       ... start/stop services based on the chkconfig status 
#

service-setup(){

   # 
   #  loop over links in /etc/init.d selecting those that point to 
   #  files owned by the current user or that have have service_name apache,
   #  for the selected files edit the script annotating with the chkconfig 2 lines
   #  that assign run levels and start/stop priority , and add the service into
   #  chkconfig management 
   #

   local msg="=== $FUNCNAME :"
   [ $(uname) == "Darwin" ] && echo $msg is a Linux thang && return 1  

   levels="345"
   start_priority="50"
   stop_priority="50"

   for item in /etc/init.d/*
   do
      if [ -L "$item" ]; then
         
         
         val=$(readlink $item)
         service_name=$(basename $item)
         script_name=$(basename $val)
         
         if [ $(id -u) == $(stat -c%u $val) ]; then
            sudo=""
         elif [ "$service_name" == "apache" ]; then
            sudo="sudo"
         else
            sudo="skip"    
         fi 
         
         echo =========== [$item] [$val] [$service_name] [$script_name] [$sudo] =========================
         
         if [ "$sudo" != "skip" ]; then
         
             chk=$(grep chkconfig $val)
             
             if [ "X$chk" == "X" ]; then
                  echo chkconfig not setup   
                  $sudo perl -pi -e "\$.==2 && printf \"# chkconfig: $levels $start_priority $stop_priority \n# description: $service_name\n\";" $val 
             else
                  echo chkconfig already setup $chk
             fi
             head -10 $val  
             
             sudo /sbin/chkconfig --add $service_name && sudo /sbin/chkconfig --list $service_name
         fi
         
	  fi	  
   done	   

}


service-list(){

   local msg="=== $FUNCNAME :"
   [ $(uname) == "Darwin" ] && echo $msg is a Linux thang && return 1  

   for item in /etc/init.d/*
   do
      if [ -L "$item" ]; then
          val=$(readlink $item)
          service_name=$(basename $item)
          script_name=$(basename $val)
          sudo /sbin/chkconfig --list $service_name
          sudo /sbin/chkconfig $service_name && echo runs at current runlevel || echo does not run at current runlevel 
      fi     
  done    
}

service-act(){

   #
   #  starts or stops services that are pointed to by symbolic links in /etc/init.d 
   #  based on the chkconfig runlevel settings and the current runlevel 
   # 

   local msg="=== $FUNCNAME :"
   [ $(uname) == "Darwin" ] && echo $msg is a Linux thang && return 1  

   for item in /etc/init.d/*
   do
      if [ -L "$item" ]; then
          val=$(readlink $item)
          service_name=$(basename $item)
          script_name=$(basename $val)
          sudo /sbin/chkconfig --list $service_name
          
          ## if the service is configured to run at the current level then start it , otherwise stop it
          sudo /sbin/chkconfig $service_name && sudo /sbin/service $service_name start || sudo /sbin/servive $service_name stop
      fi     
  done    


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





# chkconfig: 345 91 10
# description: Starts and stops the Tomcat daemon.
#   sudo /sbin/chkconfig --add tomcat
#
#         sets up a bunch of links in /etc/rc*.d/
#         presumably the priorities are just setting the start/stop order   
#
#    an informative error :
#
#[blyth@hfag env]$ /sbin/chkconfig --add tomcat
#failed to make symlink /etc/rc0.d/K10tomcat: Permission denied
#failed to make symlink /etc/rc1.d/K10tomcat: Permission denied
#failed to make symlink /etc/rc2.d/K10tomcat: Permission denied
#failed to make symlink /etc/rc3.d/S91tomcat: Permission denied
#failed to make symlink /etc/rc4.d/S91tomcat: Permission denied
#failed to make symlink /etc/rc5.d/S91tomcat: Permission denied
#failed to make symlink /etc/rc6.d/K10tomcat: Permission denied
#
#
