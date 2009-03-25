iptables-src(){ echo apache/apacheconf/iptables.bash ; }
iptables-source(){ echo $(env-home)/$(iptables-src) ; }
iptables-vi(){  vi $(iptables-source) ; }

iptables-env(){
  elocal-
}


iptables-usage(){

   cat << EOU

     $(env-wikiurl)/CMS02Firewall

     http://www.yolinux.com/TUTORIALS/LinuxTutorialIptablesNetworkGateway.html
     http://www.linuxhomenetworking.com/wiki/index.php/Quick_HOWTO_:_Ch14_:_Linux_Firewalls_Using_iptables
     http://www.cae.wisc.edu/site/public/?title=liniptables

     iptables-list
         list the chain 

     iptables-record
          leave them wallowing in working copy ... as do not want to publish them 
     
     iptables-webopen
          open port 80 to allow web access

     iptables-webclose
          close web access
          NB when testing to see if really closed there seems to be some timeout that must 
          have elapsed of order 10s

     iptables-ip   : $(iptables-ip)
          the ip address that will be used for the web{open/close}private of none is specified

     iptables-webopen-ip <ip>
          open to only the invoking/specified ip

          eg when want to restrict access ... 
                 iptables-webopen-ip 140.112.102.77
             regularize the table with 
                 iptables-webclose-ip 140.112.102.77


     iptables-webclose-ip <ip>
          open to only the invoking/specified ip
           
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
   local cmd="sudo /sbin/iptables -I $name 9 $(iptables-webaccept) $*"
   echo $msg $cmd
   eval $cmd
   iptables-list
}

iptables-webclose(){
   local msg="=== $FUNCNAME :"
   local name=$(iptables-name)
   local cmd="sudo /sbin/iptables -D $name $(iptables-webaccept) $*"
   echo $msg $cmd
   eval $cmd
   iptables-list
}


iptables-ip(){
  case $(uname) in
     Darwin) ifconfig en0  | perl -n -e 'm,inet (\S*), && print $1 '  ;;
          *) ifconfig eth0 | perl -n -e 'm,inet addr:(\S*), && print $1 '  ;;
  esac
}

iptables-webopen-ip(){
   local msg="=== $FUNCNAME :"
   local ip=${1:-$(iptables-ip)}
   [ -z "$ip" ] && echo $msg ABORT ip not specified/determined && return 1
   iptables-webopen -s $ip
}

iptables-webclose-ip(){
   local msg="=== $FUNCNAME :"
   local ip=${1:-$(iptables-ip)}
   [ -z "$ip" ] && echo $msg ABORT ip not specified/determined && return 1
   iptables-webclose -s $ip
}
