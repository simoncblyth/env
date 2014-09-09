# === func-gen- : mail/postfix fgp mail/postfix.bash fgn postfix fgh mail
postfix-src(){      echo mail/postfix.bash ; }
postfix-source(){   echo ${BASH_SOURCE:-$(env-home)/$(postfix-src)} ; }
postfix-vi(){       vi $(postfix-source) ; }
postfix-env(){      elocal- ; }
postfix-usage(){ cat << EOU

POSTFIX
=========

* http://www.techotopia.com/index.php/Configuring_an_RHEL_6_Postfix_Email_Server
* http://www.postfix.org/BASIC_CONFIGURATION_README.html
* http://www.postfix.org/STANDARD_CONFIGURATION_README.html

::

    [root@cms02 ~]# /sbin/service sendmail stop
    Shutting down sm-client:                                   [  OK  ]
    Shutting down sendmail:                                    [  OK  ]

    [root@cms02 ~]# yum install postfix
    ...
    Installed: postfix.x86_64 2:2.2.10-1.5.el4
    Complete!

    [root@cms02 ~]# rpm -ql postfix | grep cf
    /etc/postfix/main.cf
    /etc/postfix/main.cf.default
    /etc/postfix/master.cf
    /usr/share/man/man5/main.cf.5.gz


Extract from /etc/postfix/main.cf::

    # The myhostname parameter specifies the internet hostname of this
    # mail system. The default is to use the fully-qualified domain name
    # from gethostname(). $myhostname is used as a default value for many
    # other configuration parameters.

    # The mydomain parameter specifies the local internet domain name.
    # The default is to use $myhostname minus the first component.


Try operating with defaults::

    [root@cms02 ~]# /sbin/service postfix status
    master is stopped
    [root@cms02 ~]# /sbin/service postfix start
    Starting postfix:                                          [  OK  ]
    [root@cms02 ~]# 
    [root@cms02 ~]# /sbin/service postfix status
    master (pid 3928) is running...
    [root@cms02 ~]# 

Check startup without errors::

    [root@cms02 log]# tail -100 /var/log/maillog

Try sending from root and blyth::

    [root@cms02 ~]# date | mail -s "from cms02" simoncblyth@gmail.com
    [blyth@cms02 ~]$ date | mail -s "from cms02" simoncblyth@gmail.com

Test mails::
 
    delta:~ blyth$ date | mail -s "test from delta" blyth@cms02.phys.ntu.edu.tw
    [root@cms02 log]# date | mail -s "test from delta" blyth@cms02.phys.ntu.edu.tw
    [blyth@cms01 ~]$ date | mail -s "test from $(hostname)" blyth@cms02.phys.ntu.edu.tw

Local mail works::

    [blyth@cms02 ~]$ 
    You have new mail in /var/spool/mail/blyth

Remote receiving not working, try opening port 25::

     [root@cms02 ~]# IPTABLES_PORT=25 iptables-webopen


::

    [root@cms02 ~]# iptables -n --line-numbers -v -L RH-Firewall-1-INPUT
    Chain RH-Firewall-1-INPUT (2 references)
    num   pkts bytes target     prot opt in     out     source               destination         
    1      121  335K ACCEPT     all  --  lo     *       0.0.0.0/0            0.0.0.0/0           
    2    1603K  104M ACCEPT     all  --  eth0   *       0.0.0.0/0            0.0.0.0/0           
    3        0     0 ACCEPT     icmp --  *      *       0.0.0.0/0            0.0.0.0/0           icmp type 255 
    4        0     0 ACCEPT     esp  --  *      *       0.0.0.0/0            0.0.0.0/0           
    5        0     0 ACCEPT     ah   --  *      *       0.0.0.0/0            0.0.0.0/0           
    6        0     0 ACCEPT     udp  --  *      *       0.0.0.0/0            224.0.0.251         udp dpt:5353 
    7        0     0 ACCEPT     udp  --  *      *       0.0.0.0/0            0.0.0.0/0           udp dpt:631 
    8        0     0 ACCEPT     all  --  *      *       0.0.0.0/0            0.0.0.0/0           state RELATED,ESTABLISHED 
    9        0     0 ACCEPT     tcp  --  *      *       0.0.0.0/0            0.0.0.0/0           state NEW tcp dpt:22 
    10       0     0 REJECT     all  --  *      *       0.0.0.0/0            0.0.0.0/0           reject-with icmp-host-prohibited 

    [root@cms02 ~]# iptables -I RH-Firewall-1-INPUT 1 -i eth0 -p tcp --sport 25 -m state --state ESTABLISHED -j ACCEPT
    [root@cms02 ~]# 
    [root@cms02 ~]# 
    [root@cms02 ~]# iptables -n --line-numbers -v -L RH-Firewall-1-INPUT
    Chain RH-Firewall-1-INPUT (2 references)
    num   pkts bytes target     prot opt in     out     source               destination         
    1        0     0 ACCEPT     tcp  --  eth0   *       0.0.0.0/0            0.0.0.0/0           tcp spt:25 state ESTABLISHED 
    2      121  335K ACCEPT     all  --  lo     *       0.0.0.0/0            0.0.0.0/0           
    3    1607K  104M ACCEPT     all  --  eth0   *       0.0.0.0/0            0.0.0.0/0           
    4        0     0 ACCEPT     icmp --  *      *       0.0.0.0/0            0.0.0.0/0           icmp type 255 
    5        0     0 ACCEPT     esp  --  *      *       0.0.0.0/0            0.0.0.0/0           
    6        0     0 ACCEPT     ah   --  *      *       0.0.0.0/0            0.0.0.0/0           
    7        0     0 ACCEPT     udp  --  *      *       0.0.0.0/0            224.0.0.251         udp dpt:5353 
    8        0     0 ACCEPT     udp  --  *      *       0.0.0.0/0            0.0.0.0/0           udp dpt:631 
    9        0     0 ACCEPT     all  --  *      *       0.0.0.0/0            0.0.0.0/0           state RELATED,ESTABLISHED 
    10       0     0 ACCEPT     tcp  --  *      *       0.0.0.0/0            0.0.0.0/0           state NEW tcp dpt:22 
    11       0     0 REJECT     all  --  *      *       0.0.0.0/0            0.0.0.0/0           reject-with icmp-host-prohibited 
    [root@cms02 ~]# 


Not managing to let in any counts until, use *-m tcp -p tcp* 

    iptables -I RH-Firewall-1-INPUT 1 -i eth0  -m state --state NEW -m tcp -p tcp --dport 25 -j ACCEPT


EOU
}
postfix-dir(){ echo $(local-base)/env/mail/mail-postfix ; }
postfix-cd(){  cd $(postfix-dir); }
postfix-mate(){ mate $(postfix-dir) ; }
postfix-get(){
   local dir=$(dirname $(postfix-dir)) &&  mkdir -p $dir && cd $dir

}
