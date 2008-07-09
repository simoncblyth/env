


iptables-usage(){

   cat << EOU

http://www.yolinux.com/TUTORIALS/LinuxTutorialIptablesNetworkGateway.html
http://www.linuxhomenetworking.com/wiki/index.php/Quick_HOWTO_:_Ch14_:_Linux_Firewalls_Using_iptables


     iptables-record
          leave them wallowing in working copy ... as do not want to publish them 
     
     iptables-webopen
           
       TODO:
           make this persistent across reboots ...  


EOU

}


iptables-dir(){  echo $ENV_HOME/apache/apacheconf/iptables ; }

iptables-name(){ echo RH-Firewall-1-INPUT ; }

iptables-record(){

   local dir=$(iptables-dir)
   mkdir -p $dir 
   cd $dir   
   local name=$(iptables-name)
   sudo iptables --line-numbers --list $name   > $name.list 
   sudo iptables-save > iptables-save.txt

}


iptables-webopen(){

   local name=$(iptables-name)
   sudo iptables -I $name 9 -p tcp -i eth0 --dport 80 --sport 1024:65535 -m state --state NEW -j ACCEPT

}




