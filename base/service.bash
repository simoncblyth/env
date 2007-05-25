
#
# service-setup     ... do the chkconfig script annotation
# service-list      ... list the chkconfig status
# service-act       ... start/stop services based on the chkconfig status 
#

service-setup(){

   # 
   #  loop over links in /etc/init.d selecting those that point to 
   #  files owned by the current user or that have have basename apachectl,
   #  for the selected files edit the script annotating with the chkconfig 2 lines
   #  that assign run levels and start/stop priority , and add the service into
   #  chkconfig management 
   #

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