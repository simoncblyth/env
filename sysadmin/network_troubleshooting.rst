Network Troubleshooting
=========================

Overview
----------

The problem that inspired creation of this page turned out to 
be caused by the cms02 ethernet cable being plugged into the wrong NIC.
How that happened following the powercut over the weekend remains unknown, 
but checking apache logs suggests it happened at about 10:26 
on Monday morning.

RESOLVED as due to accidental miscabling
------------------------------------------

Lessons learned, regarding appropriate troubleshooting sequence:

#. **1st priority** check apache and other logs to **piece together the sequence of events** 
   (its kinda uncomfortable doing this on the console in server room, but its necessary)

#. check which NIC the cable is attached to first, label the correct one


Sequence of events
---------------------------

The reason it took a while to work this out is that the other NIC 
worked partially (IPV6 traffic was moving).


::

    01/Aug/2014:22:20:41 +0800   last apache access before power outage on Friday evening 
    04/Aug/2014:08:11:07 +0800   first apache access after power regained and system rebooted on Monday morning
    04/Aug/2014:10:25:28 +0800   last apache access at ~10:25 Monday  

::

    blyth    tty1                          Tue Aug  5 11:14 - 11:16  (00:01)    
    reboot   system boot  2.6.9-78.0.22.EL Mon Aug  4 11:46         (2+00:01)   
    root     tty1                          Mon Aug  4 11:23 - 11:36  (00:12)    
    reboot   system boot  2.6.9-78.0.22.EL Mon Aug  4 11:23         (2+00:24)   
    root     tty1                          Mon Aug  4 11:21 - down   (00:00)    
    blyth    tty1                          Mon Aug  4 11:12 - 11:12  (00:00)    
    reboot   system boot  2.6.9-78.0.22.EL Mon Aug  4 08:10          (03:11)     


Network Troubleshooting Refs
-------------------------------

* http://unix.stackexchange.com/questions/50098/linux-network-troubleshooting-and-debugging
* http://unix.stackexchange.com/questions/83273/how-to-diagnose-faulty-onboard-network-adapter
* http://www.linuxhomenetworking.com/wiki/index.php/Quick_HOWTO_%3a_Ch04_%3a_Simple_Network_Troubleshooting#.U-Gd21YWe2w


pinging
---------

#. Pinging 127.0.0.1 (loopback) tests that your TCP/IP stack isn't corrupt. 
#. Pinging your ip address address (e.g., 192.168.1.100) tests if your NIC is functioning.
#. Pinging your gateway tests if your LAN is working. 

   * as root: `netstat -nr` to findout the gateway

#. Pinging an external address tests if routing and your internet connection are working.


Troubleshooting Tools
-----------------------

ifconfig
~~~~~~~~~~~

#. The ifconfig command without any arguments gives you all the active interfaces on your system. 

   * Interfaces will not appear if they are shut down:
   * Interfaces will appear if they are activated, but have no link. 
   * look for **UP** in the ifconfig output 
   * An interface with a 169.254.x.x address signifies a failure to communicate with the DHCP server. 


#. Bring interfaces up/down with `ifconfig eth0 up`


ethtool
^^^^^^^^^

::

    [root@cms01 blyth]# which ethtool
    /sbin/ethtool

    [root@cms01 blyth]# ethtool eth0
    Settings for eth0:
        Supported ports: [ TP MII ]
        Supported link modes:   10baseT/Half 10baseT/Full 
                                100baseT/Half 100baseT/Full 
        Supports auto-negotiation: Yes
        Advertised link modes:  10baseT/Half 10baseT/Full 
                                100baseT/Half 100baseT/Full 
        Advertised auto-negotiation: Yes
        Speed: 100Mb/s
        Duplex: Full
        Port: MII
        PHYAD: 24
        Transceiver: internal
        Auto-negotiation: on
        Current message level: 0x00000001 (1)
        Link detected: yes


::

    [root@cms01 blyth]# ethtool eth1
    Settings for eth1:
        Supported ports: [ TP ]
        Supported link modes:   10baseT/Half 10baseT/Full 
                                100baseT/Half 100baseT/Full 
                                1000baseT/Full 
        Supports auto-negotiation: Yes
        Advertised link modes:  10baseT/Half 10baseT/Full 
                                100baseT/Half 100baseT/Full 
                                1000baseT/Full 
        Advertised auto-negotiation: Yes
        Speed: Unknown! (65535)
        Duplex: Unknown! (255)
        Port: Twisted Pair
        PHYAD: 0
        Transceiver: internal
        Auto-negotiation: on
        Supports Wake-on: d
        Wake-on: d
        Current message level: 0x00000007 (7)
        Link detected: no

::

    [root@cms01 blyth]# ethtool -S eth0
    NIC statistics:
         tx_deferred: 0
         tx_multiple_collisions: 0
         rx_bad_ssd: 26


::

    [root@cms01 blyth]# ethtool -S eth1
    NIC statistics:
         rx_packets: 0
         tx_packets: 0
         rx_bytes: 0
         tx_bytes: 0
         ...             # many other stats, all 0




mii-tool
~~~~~~~~~~

#. reveals that cms02 has inet6 interface that succeeds to come up ?


::

    [root@cms01 blyth]# mii-tool -v
    eth0: negotiated 100baseTx-FD, link ok
      product info: vendor 00:10:5a, model 0 rev 0
      basic mode:   autonegotiation enabled
      basic status: autonegotiation complete, link ok
      capabilities: 100baseTx-FD 100baseTx-HD 10baseT-FD 10baseT-HD
      advertising:  100baseTx-FD 100baseTx-HD 10baseT-FD 10baseT-HD flow-control
      link partner: 100baseTx-FD 100baseTx-HD 10baseT-FD 10baseT-HD
    eth1: no link
      product info: vendor 00:50:43, model 2 rev 3
      basic mode:   autonegotiation enabled
      basic status: no link
      capabilities: 100baseTx-FD 100baseTx-HD 10baseT-FD 10baseT-HD
      advertising:  100baseTx-FD 100baseTx-HD 10baseT-FD 10baseT-HD flow-control
    [root@cms01 blyth]# 


arp
~~~~

* http://dougvitale.wordpress.com/2011/12/11/troubleshooting-faulty-network-connectivity-part-2-essential-network-commands/#arp

arp command lets you view and manage the Address Resolution Protocol (ARP) cache. 

As DNS translates between host names and IP addresses, ARP translates between
MAC addresses (Layer 2) and IP addresses (Layer 3). When a host attempts to
communicate with another host on the same subnet, it must first know the
destination host’s MAC address. If there is no entry in the sending host’s ARP
cache for the destination MAC address, ARP sends out a broadcast (to all hosts
in the subnet) asking the host with the target IP address to send back its MAC
address. These IP-to-MAC mappings build up in the ARP cache which the arp
command lets you view and modify.

IP address to MAC address cache, was empty on cms02 had just the gateway on cms01::

    [root@cms01 blyth]# arp -a


dmesg
~~~~~~~

The boot output from miscabled cms02 had::

    ADDRCONF(NETDEV_UP): eth0: link is not ready



netstat
~~~~~~~~

Activity on the interfaces::

    [root@cms01 blyth]# netstat -i
    Kernel Interface table
    Iface       MTU Met    RX-OK RX-ERR RX-DRP RX-OVR    TX-OK TX-ERR TX-DRP TX-OVR Flg
    eth0       1500   0 14542429      0      0      1  3798504      0      0      0 BMRU
    eth1       1500   0        0      0      0      0        0      0      0      0 BMU
    lo        16436   0     1468      0      0      0     1468      0      0      0 LRU

Find the gateway::


    [root@cms01 blyth]# netstat -nr
    Kernel IP routing table
    Destination     Gateway         Genmask         Flags   MSS Window  irtt Iface
    140.112.101.0   0.0.0.0         255.255.255.0   U         0 0          0 eth0
    169.254.0.0     0.0.0.0         255.255.0.0     U         0 0          0 eth1
    192.168.0.0     0.0.0.0         255.255.0.0     U         0 0          0 eth1
    0.0.0.0         140.112.101.254 0.0.0.0         UG        0 0          0 eth0

Ping the gateway, works on cms01 got `Destination Unreachable` on miscabled cms02::

    [root@cms01 blyth]# ping 140.112.101.254
    PING 140.112.101.254 (140.112.101.254) 56(84) bytes of data.
    64 bytes from 140.112.101.254: icmp_seq=0 ttl=255 time=0.657 ms
    64 bytes from 140.112.101.254: icmp_seq=1 ttl=255 time=5.24 ms
    64 bytes from 140.112.101.254: icmp_seq=2 ttl=255 time=0.593 ms

    --- 140.112.101.254 ping statistics ---
    3 packets transmitted, 3 received, 0% packet loss, time 2000ms
    rtt min/avg/max/mdev = 0.593/2.164/5.243/2.177 ms, pipe 2


Checking configuration
------------------------

::

    [root@cms01 blyth]# cat /etc/sysconfig/network-scripts/ifcfg-eth0
 

Stop/start network service
----------------------------

::

   service network restart




cms02 inet6 redherring
------------------------

Try connecting cms02 via inet6 interface::


    [root@cms01 blyth]# ping6 -I eth1 fe80::207:e9ff:fe13:ea50/64
    unknown host

    [root@cms01 blyth]# ping6 -I eth1 fe80::207:e9ff:fe13:ea50
    connect: Network is unreachable


* http://www.ipv6.leeds.ac.uk/Docs/ipv6_linux_tips.txt


::

    [root@cms01 blyth]# ping6 ::1   
    PING ::1(::1) 56 data bytes
    64 bytes from ::1: icmp_seq=0 ttl=64 time=0.062 ms
    64 bytes from ::1: icmp_seq=1 ttl=64 time=0.018 ms
    64 bytes from ::1: icmp_seq=2 ttl=64 time=0.019 ms


    [root@cms01 blyth]# route -A inet6
    Kernel IPv6 routing table
    Destination                                 Next Hop                                Flags Metric Ref    Use Iface




Checklist
-----------

* http://dougvitale.wordpress.com/2011/11/28/troubleshooting-faulty-network-connectivity-part-1/


If you can ping both the loopback address and your own IP address but not hosts
in the local subnet, try to clear out the ARP cache and reload it. This can be
done by using the Arp utility on the command line interface (CLI). First
display the cache entries with the arp -a or arp -g commands. Delete the
entries with arp -d <IP address>. For the full list of options available for
the arp command, go here.

Check Config
--------------

* https://access.redhat.com/documentation/en-US/Red_Hat_Enterprise_Linux/6/html/Deployment_Guide/s1-networkscripts-interfaces.html


::

    [blyth@cms01 network-scripts]$ cat /etc/sysconfig/network
    [blyth@cms01 network-scripts]$ cat ifcfg-eth0

The global directive NETWORKING_IPV6 is required in the /etc/sysconfig/network
conf file to globally enable IPv6 static, DHCP, or autoconf configuration.
Refer to Section D.1.13, “/etc/sysconfig/network”


* https://bugzilla.redhat.com/show_bug.cgi?id=125587

::

    [blyth@cms01 network-scripts]$ rpm -qf /etc/sysconfig/network
    file /etc/sysconfig/network is not owned by any package


* http://www.windowsnetworking.com/articles-tutorials/trouble/TCPIP-Troubleshooting-Structured-Approach-Part1.html


