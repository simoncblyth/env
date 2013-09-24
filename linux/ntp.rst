Network Time Protocol
======================

* `http://www.linuxhomenetworking.com/wiki/index.php/Quick_HOWTO_:_Ch24_:_The_NTP_Server`


Update the datetime
---------------------

Time was back in 2000, the below ntpdate got back in the ballpark, but timezone was off::

    [blyth@hfag blyth]$ sudo /usr/sbin/ntpdate -u ntp.ucsd.edu
     8 Sep 20:23:22 ntpdate[2805]: step time server 132.239.1.6 offset 429053400.504246 sec
    [blyth@hfag blyth]$ date
    Sun Sep  8 20:23:29 GMT+8 2013
    [blyth@hfag blyth]$ sudo /usr/sbin/ntpdate -u ntp.ucsd.edu
    Password:
     8 Sep 20:23:53 ntpdate[2808]: adjust time server 132.239.1.6 offset 0.000532 sec
    [blyth@hfag blyth]$ date
    Sun Sep  8 20:24:03 GMT+8 2013
    [blyth@hfag blyth]


Timezone setup 
----------------

* http://www.hypexr.org/linux_date_time_help.php

Creating a symbolic link from `/etc/localtime` to the zoneinfo file succeeded to get it straight.::

    [blyth@hfag blyth]$ sudo rm  /etc/localtime 
    [blyth@hfag blyth]$ sudo ln -s /usr/share/zoneinfo/Asia/Taipei /etc/localtime 
    [blyth@hfag blyth]$ date
    Mon Sep  9 12:35:34 CST 2013


Config
--------

The ntp server to connect to was gleaned from the conf. 

* `/etc/ntp.conf`

For the docs etc..
--------------------

::

    [blyth@hfag blyth]$ rpm -ql ntp
    /etc/ntp
    /etc/ntp.conf
    /etc/ntp/keys
    /etc/ntp/step-tickers
    /etc/rc.d/init.d/ntpd
    /etc/sysconfig/ntpd
    /usr/bin/ntpstat
    /usr/sbin/ntp-genkeys
    /usr/sbin/ntp-wait
    /usr/sbin/ntpd
    /usr/sbin/ntpdate
    /usr/sbin/ntpdc
    /usr/sbin/ntpq
    /usr/sbin/ntptime
    /usr/sbin/ntptimeset
    /usr/sbin/ntptrace
    /usr/sbin/tickadj
    /usr/share/doc/ntp-4.1.2
    /usr/share/doc/ntp-4.1.2/NEWS
    /usr/share/doc/ntp-4.1.2/Oncore-SHMEM.htm
    /usr/share/doc/ntp-4.1.2/TODO
    /usr/share/doc/ntp-4.1.2/accopt.htm
    /usr/share/doc/ntp-4.1.2/assoc.htm
    /usr/share/doc/ntp-4.1.2/audio.htm
    ...




