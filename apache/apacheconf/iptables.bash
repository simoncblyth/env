iptables-src(){ echo apache/apacheconf/iptables.bash ; }
iptables-source(){ echo $(env-home)/$(iptables-src) ; }
iptables-vi(){  vi $(iptables-source) ; }

iptables-env(){
  elocal-
}
iptables-ini(){    sudo $(iptables-inipath)  $*  ; }
iptables-inipath(){ echo /etc/init.d/iptables    ; }
iptables-syspath(){ echo /etc/sysconfig/iptables ; }


iptables-smtpopen-notes(){ cat << EON

http://stackoverflow.com/questions/10670742/how-to-allow-mail-through-iptables


2. A hint from the trenches: when you're debugging iptables, it's often helpful
to -Insert and -Append log messages at the beginning and end of each chain,
then clear the counters, and run an experiment. (In your case, issue the mail
command.) Then check the counters and logs to understand how the packet(s)
migrated through the chains and where they may have been dropped.


[root@cms02 ~]# service iptables restart
Flushing firewall rules:                                   [  OK  ]
Setting chains to policy ACCEPT: filter                    [  OK  ]
Unloading iptables modules:                                [  OK  ]
Applying iptables firewall rules:                          [  OK  ]
[root@cms02 ~]# 
[root@cms02 ~]# 
[root@cms02 ~]# service iptables status
Table: filter
Chain INPUT (policy ACCEPT)
target     prot opt source               destination         
RH-Firewall-1-INPUT  all  --  0.0.0.0/0            0.0.0.0/0           

Chain FORWARD (policy ACCEPT)
target     prot opt source               destination         
RH-Firewall-1-INPUT  all  --  0.0.0.0/0            0.0.0.0/0           

Chain OUTPUT (policy ACCEPT)
target     prot opt source               destination         

Chain RH-Firewall-1-INPUT (2 references)
target     prot opt source               destination         
ACCEPT     all  --  0.0.0.0/0            0.0.0.0/0           
ACCEPT     all  --  0.0.0.0/0            0.0.0.0/0           
ACCEPT     icmp --  0.0.0.0/0            0.0.0.0/0           icmp type 255 
ACCEPT     esp  --  0.0.0.0/0            0.0.0.0/0           
ACCEPT     ah   --  0.0.0.0/0            0.0.0.0/0           
ACCEPT     udp  --  0.0.0.0/0            224.0.0.251         udp dpt:5353 
ACCEPT     udp  --  0.0.0.0/0            0.0.0.0/0           udp dpt:631 
ACCEPT     all  --  0.0.0.0/0            0.0.0.0/0           state RELATED,ESTABLISHED 
ACCEPT     tcp  --  0.0.0.0/0            0.0.0.0/0           state NEW tcp dpt:22 
REJECT     all  --  0.0.0.0/0            0.0.0.0/0           reject-with icmp-host-prohibited 





EON
}


iptables-input(){
  iptables -n -v --line-numbers -L $(iptables-name)
}


iptables-usage(){

   cat << EOU

     $(env-wikiurl)/CMS02Firewall
     $(env-wikiurl)/IPTables

     http://www.yolinux.com/TUTORIALS/LinuxTutorialIptablesNetworkGateway.html
     http://www.linuxhomenetworking.com/wiki/index.php/Quick_HOWTO_:_Ch14_:_Linux_Firewalls_Using_iptables
     http://www.cae.wisc.edu/site/public/?title=liniptables

     http://www.cyberciti.biz/faq/how-do-i-save-iptables-rules-or-settings/

     iptables-inipath : $(iptables-inipath)
     iptables-syspath : $(iptables-syspath)

          

     iptables-list
         list the chain 

     iptables-record
          leave them wallowing in working copy ... as do not want to publish them 
     
     iptables-port   : $(iptables-port)
          port to control, override default with IPTABLES_PORT : $IPTABLES_PORT

     iptables-webopen
          open port $(iptables-port) to allow web access

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
         
     == persisting iptables on C ==
 
          make iptables settings persistent across reboots ...  
         > [blyth@cms01 log]$ iptables-persist
         > sudo /sbin/service iptables save
         > Saving firewall rules to /etc/sysconfig/iptables:          [  OK  ]
     
         > [blyth@cms01 log]$ sudo service iptables
         > Usage: /etc/init.d/iptables {start|stop|restart|condrestart|status|panic|save}
 
     == persisting iptables on C2 ==

         [blyth@cms02 ~]$ iptables-ini
         Usage: /etc/init.d/iptables {start|stop|restart|condrestart|status|panic|save}
         [blyth@cms02 ~]$ iptables-ini save
         Saving firewall rules to /etc/sysconfig/iptables:          [  OK  ]
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

iptables-port(){ echo ${IPTABLES_PORT:-80} ; }

iptables-webaccept(){
  echo -p tcp -i eth0 --dport $(iptables-port) --sport 1024:65535 -m state --state NEW -j ACCEPT
}

iptables-list(){
   local msg="=== $FUNCNAME :"
   local cmd="sudo /sbin/iptables --line-numbers --list $(iptables-name)"
   echo $msg $cmd
   eval $cmd
}

iptables-undrop-ip(){
   local msg="=== $FUNCNAME :"
   local ip=$1
   local cmd="sudo /sbin/iptables -D INPUT -s $ip -j DROP"
   echo $msg $cmd
   eval $cmd
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
     Darwin) /sbin/ifconfig en0  | perl -n -e 'm,inet (\S*), && print $1 '  ;;
          *) /sbin/ifconfig eth0 | perl -n -e 'm,inet addr:(\S*), && print $1 '  ;;
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

iptables-open(){
   local msg="=== $FUNCNAME :"
   local port=${1:-6060}
   local tag=G
   local cmd="IPTABLES_PORT=${1:-8080} iptables-webopen-ip $(local-tag2ip $tag) "
   echo $msg $cmd
   eval $cmd
}

iptables-setup-node(){
   local msg="=== $FUNCNAME :"
   if [ "$(local-nodetag)" == "C" ]; then
       IPTABLES_PORT=$(local-port mysql) iptables-webopen-ip $(local-tag2ip C2)   
       IPTABLES_PORT=$(local-port rabbitmq) iptables-webopen  
       IPTABLES_PORT=$(local-port slave-lighttpd) iptables-webopen        
   elif [ "$(local-nodetag)" == "C2" ]; then
       IPTABLES_PORT=$(local-port apache) iptables-webopen  
   else 
       echo $msg no setup defined for this node $(local-nodetag)  ... && return 0 
   fi  
   iptables-persist 
}


