iptables-src(){ echo apache/apacheconf/iptables.bash ; }
iptables-source(){ echo $(env-home)/$(iptables-src) ; }
iptables-vi(){  vi $(iptables-source) ; }

iptables-env(){
  elocal-
}


iptables-usage(){

   cat << EOU

     $(env-wikiurl CMS02Firewall)

     http://www.yolinux.com/TUTORIALS/LinuxTutorialIptablesNetworkGateway.html
     http://www.linuxhomenetworking.com/wiki/index.php/Quick_HOWTO_:_Ch14_:_Linux_Firewalls_Using_iptables
     http://www.cae.wisc.edu/site/public/?title=liniptables

     iptables-record
          leave them wallowing in working copy ... as do not want to publish them 
     
     iptables-webopen
          open port 80 to allow web access

     iptables-webopenprivate <ip>
          open port 80 to allow web access

     iptables-webclose
          close web access
           
     iptables-persist
          
          make iptables settings persistent across reboots ...  
         > [blyth@cms01 log]$ iptables-persist
         > sudo /sbin/service iptables save
         > Saving firewall rules to /etc/sysconfig/iptables:          [  OK  
     
         > [blyth@cms01 log]$ sudo service iptables
         > Usage: /etc/init.d/iptables {start|stop|restart|condrestart|status|panic|save}


EOU

}


iptables-persist(){
  local cmd="sudo /sbin/service iptables save"
  echo $cmd
  eval $cmd
}


iptables-ls(){   ls -l $(iptables-dir) ; }

iptables-dir(){  echo $ENV_HOME/apache/apacheconf/iptables ; }

iptables-name(){ echo RH-Firewall-1-INPUT ; }

iptables-record(){

   local dir=$(iptables-dir)
   mkdir -p $dir 
   cd $dir   
   local name=$(iptables-name)
   sudo /sbin/iptables --line-numbers --list $name   > $name.list 
   sudo /sbin/iptables-save > iptables-save.txt

}

iptables-webaccept(){
  echo -p tcp -i eth0 --dport 80 --sport 1024:65535 -m state --state NEW -j ACCEPT
}

iptables-list(){
   sudo /sbin/iptables --line-numbers --list $(iptables-name)
}

iptables-webopen(){
   local msg="=== $FUNCNAME :"
   local name=$(iptables-name)
   local cmd="sudo /sbin/iptables -I $name 9 $(iptables-webaccept)"
   echo $msg $cmd
   eval $cmd
   iptables-list
}

iptables-webclose(){
   local msg="=== $FUNCNAME :"
   local name=$(iptables-name)
   local cmd="sudo /sbin/iptables -D $name $(iptables-webaccept)"
   echo $msg $cmd
   eval $cmd
   iptables-list
}

iptables-webopenprivate(){
   echo -n
}

