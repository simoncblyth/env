CMS02
======


July 30, 2015 Manual Stop prior to powercut
---------------------------------------------

Note that commandline is slow.::

    Last login: Thu Jul 30 15:16:29 on ttys011
    simon:~ blyth$ vin
    simon:~ blyth$ ssh C2
    Enter passphrase for key '/Users/blyth/.ssh/id_rsa': 
    Last login: Thu Jul 30 15:13:19 2015 from simon.phys.ntu.edu.tw
    [blyth@cms02 ~]$ 
    [blyth@cms02 ~]$ sudo /sbin/shutdown -h now "goin down"
    Password:

    Broadcast message from root (pts/0) (Thu Jul 30 15:35:47 2015):

    goin down 
    The system is going down for system halt NOW!










Oct 14, 2014
-------------

Its up, but cannot ssh in and pinging has intermittent timeouts::

    delta:swift blyth$ ping cms02.phys.ntu.edu.tw
    PING cms02.phys.ntu.edu.tw (140.112.101.191): 56 data bytes
    64 bytes from 140.112.101.191: icmp_seq=0 ttl=62 time=1.940 ms
    64 bytes from 140.112.101.191: icmp_seq=1 ttl=62 time=2.330 ms
    Request timeout for icmp_seq 2
    64 bytes from 140.112.101.191: icmp_seq=3 ttl=62 time=2.093 ms
    Request timeout for icmp_seq 4
    64 bytes from 140.112.101.191: icmp_seq=5 ttl=62 time=2.071 ms
    64 bytes from 140.112.101.191: icmp_seq=6 ttl=62 time=2.137 ms
    64 bytes from 140.112.101.191: icmp_seq=7 ttl=62 time=13.915 ms
    Request timeout for icmp_seq 8
    64 bytes from 140.112.101.191: icmp_seq=9 ttl=62 time=2.143 ms
    64 bytes from 140.112.101.191: icmp_seq=10 ttl=62 time=5.739 ms
    64 bytes from 140.112.101.191: icmp_seq=11 ttl=62 time=2.199 ms
    ^C
    --- cms02.phys.ntu.edu.tw ping statistics ---
    12 packets transmitted, 9 packets received, 25.0% packet loss
    round-trip min/avg/max/stddev = 1.940/3.841/13.915/3.737 ms
    delta:swift blyth$ 



Sep 4, 2014
-------------

#. googlebot is worst offender, so in google webmaster tools limit crawl rate to 0.01 per second 



Ongoing::

    [root@cms02 logs]# DAY=02/Sep/2014 apachehits.py
    cat access_log|grep 02/Sep/2014:00|wc -l 1515
    cat access_log|grep 02/Sep/2014:01|wc -l 1405
    cat access_log|grep 02/Sep/2014:02|wc -l 1362
    cat access_log|grep 02/Sep/2014:03|wc -l 1274
    cat access_log|grep 02/Sep/2014:04|wc -l 1219
    cat access_log|grep 02/Sep/2014:05|wc -l 1305
    cat access_log|grep 02/Sep/2014:06|wc -l 1418
    cat access_log|grep 02/Sep/2014:07|wc -l 1577
    cat access_log|grep 02/Sep/2014:08|wc -l 1380
    cat access_log|grep 02/Sep/2014:09|wc -l 1243
    cat access_log|grep 02/Sep/2014:10|wc -l 248
    cat access_log|grep 02/Sep/2014:11|wc -l 244
    cat access_log|grep 02/Sep/2014:12|wc -l 342
    cat access_log|grep 02/Sep/2014:13|wc -l 365
    cat access_log|grep 02/Sep/2014:14|wc -l 377
    cat access_log|grep 02/Sep/2014:15|wc -l 390
    cat access_log|grep 02/Sep/2014:16|wc -l 423
    cat access_log|grep 02/Sep/2014:17|wc -l 550
    cat access_log|grep 02/Sep/2014:18|wc -l 411
    cat access_log|grep 02/Sep/2014:19|wc -l 412
    cat access_log|grep 02/Sep/2014:20|wc -l 478
    cat access_log|grep 02/Sep/2014:21|wc -l 554
    cat access_log|grep 02/Sep/2014:22|wc -l 429
    cat access_log|grep 02/Sep/2014:23|wc -l 682
    [root@cms02 logs]# 
    [root@cms02 logs]# 

    [root@cms02 logs]# DAY=03/Sep/2014 apachehits.py
    cat access_log|grep 03/Sep/2014:00|wc -l 767
    cat access_log|grep 03/Sep/2014:01|wc -l 746
    cat access_log|grep 03/Sep/2014:02|wc -l 739
    cat access_log|grep 03/Sep/2014:03|wc -l 788
    cat access_log|grep 03/Sep/2014:04|wc -l 845
    cat access_log|grep 03/Sep/2014:05|wc -l 3
    cat access_log|grep 03/Sep/2014:06|wc -l 0
    cat access_log|grep 03/Sep/2014:07|wc -l 0
    cat access_log|grep 03/Sep/2014:08|wc -l 1
    cat access_log|grep 03/Sep/2014:09|wc -l 0
    cat access_log|grep 03/Sep/2014:10|wc -l 0
    cat access_log|grep 03/Sep/2014:11|wc -l 0
    cat access_log|grep 03/Sep/2014:12|wc -l 0
    cat access_log|grep 03/Sep/2014:13|wc -l 0
    cat access_log|grep 03/Sep/2014:14|wc -l 0
    cat access_log|grep 03/Sep/2014:15|wc -l 0
    cat access_log|grep 03/Sep/2014:16|wc -l 41
    cat access_log|grep 03/Sep/2014:17|wc -l 690
    cat access_log|grep 03/Sep/2014:18|wc -l 398
    cat access_log|grep 03/Sep/2014:19|wc -l 0
    cat access_log|grep 03/Sep/2014:20|wc -l 0
    cat access_log|grep 03/Sep/2014:21|wc -l 0
    cat access_log|grep 03/Sep/2014:22|wc -l 0
    cat access_log|grep 03/Sep/2014:23|wc -l 0

    [root@cms02 logs]# DAY=04/Sep/2014 apachehits.py
    cat access_log|grep 04/Sep/2014:00|wc -l 1
    cat access_log|grep 04/Sep/2014:01|wc -l 0
    cat access_log|grep 04/Sep/2014:02|wc -l 0
    cat access_log|grep 04/Sep/2014:03|wc -l 2
    cat access_log|grep 04/Sep/2014:04|wc -l 1
    cat access_log|grep 04/Sep/2014:05|wc -l 0
    cat access_log|grep 04/Sep/2014:06|wc -l 0
    cat access_log|grep 04/Sep/2014:07|wc -l 0
    cat access_log|grep 04/Sep/2014:08|wc -l 2
    cat access_log|grep 04/Sep/2014:09|wc -l 0
    cat access_log|grep 04/Sep/2014:10|wc -l 0
    cat access_log|grep 04/Sep/2014:11|wc -l 0
    cat access_log|grep 04/Sep/2014:12|wc -l 0
    cat access_log|grep 04/Sep/2014:13|wc -l 0
    cat access_log|grep 04/Sep/2014:14|wc -l 0
    cat access_log|grep 04/Sep/2014:15|wc -l 0
    cat access_log|grep 04/Sep/2014:16|wc -l 122
    cat access_log|grep 04/Sep/2014:17|wc -l 0
    cat access_log|grep 04/Sep/2014:18|wc -l 0
    cat access_log|grep 04/Sep/2014:19|wc -l 0
    cat access_log|grep 04/Sep/2014:20|wc -l 0
    cat access_log|grep 04/Sep/2014:21|wc -l 0
    cat access_log|grep 04/Sep/2014:22|wc -l 0
    cat access_log|grep 04/Sep/2014:23|wc -l 0
    [root@cms02 logs]# 




Sep 1, 2014
------------

Same again. Continuation of the same attack.

Only came back up for a few hours::

    IP=66.249.76.74 DAY=29/Aug/2014 apachehits.py

    [root@cms02 logs]# DAY=29/Aug/2014 apachehits.py
    cat access_log|grep 29/Aug/2014:00|wc -l 0
    cat access_log|grep 29/Aug/2014:01|wc -l 0
    cat access_log|grep 29/Aug/2014:02|wc -l 2
    cat access_log|grep 29/Aug/2014:03|wc -l 0
    cat access_log|grep 29/Aug/2014:04|wc -l 0
    cat access_log|grep 29/Aug/2014:05|wc -l 3
    cat access_log|grep 29/Aug/2014:06|wc -l 0
    cat access_log|grep 29/Aug/2014:07|wc -l 0
    cat access_log|grep 29/Aug/2014:08|wc -l 3
    cat access_log|grep 29/Aug/2014:09|wc -l 0
    cat access_log|grep 29/Aug/2014:10|wc -l 0
    cat access_log|grep 29/Aug/2014:11|wc -l 3
    cat access_log|grep 29/Aug/2014:12|wc -l 634
    cat access_log|grep 29/Aug/2014:13|wc -l 356
    cat access_log|grep 29/Aug/2014:14|wc -l 131
    cat access_log|grep 29/Aug/2014:15|wc -l 0
    cat access_log|grep 29/Aug/2014:16|wc -l 0
    cat access_log|grep 29/Aug/2014:17|wc -l 0
    cat access_log|grep 29/Aug/2014:18|wc -l 0
    cat access_log|grep 29/Aug/2014:19|wc -l 0
    cat access_log|grep 29/Aug/2014:20|wc -l 0
    cat access_log|grep 29/Aug/2014:21|wc -l 0
    cat access_log|grep 29/Aug/2014:22|wc -l 0
    cat access_log|grep 29/Aug/2014:23|wc -l 0




Whois indicates google is culprit::

    [root@cms02 logs]# IP=66.249.76.74 DAY=29/Aug/2014 apachehits.py
    ...
    cat access_log|grep ^66.249.76.74|grep 29/Aug/2014:11|wc -l 0
    cat access_log|grep ^66.249.76.74|grep 29/Aug/2014:12|wc -l 424
    cat access_log|grep ^66.249.76.74|grep 29/Aug/2014:13|wc -l 166
    cat access_log|grep ^66.249.76.74|grep 29/Aug/2014:14|wc -l 33
    cat access_log|grep ^66.249.76.74|grep 29/Aug/2014:15|wc -l 0
    ...




Aug 29, 2014
-------------

* noted cms02 apache/svn not responding yesterday evening Aug 28
* this morning: can ping, but cannot ssh or web access, reboot regains access
* looks like another robot attack killing apache


::

    [root@cms02 logs]# DAY=27/Aug/2014 apachehits.py
    cat access_log|grep 27/Aug/2014:00|wc -l 8
    cat access_log|grep 27/Aug/2014:01|wc -l 0
    cat access_log|grep 27/Aug/2014:02|wc -l 2
    cat access_log|grep 27/Aug/2014:03|wc -l 2
    cat access_log|grep 27/Aug/2014:04|wc -l 0
    cat access_log|grep 27/Aug/2014:05|wc -l 1
    cat access_log|grep 27/Aug/2014:06|wc -l 1
    cat access_log|grep 27/Aug/2014:07|wc -l 0
    cat access_log|grep 27/Aug/2014:08|wc -l 0
    cat access_log|grep 27/Aug/2014:09|wc -l 4
    cat access_log|grep 27/Aug/2014:10|wc -l 17
    cat access_log|grep 27/Aug/2014:11|wc -l 8
    cat access_log|grep 27/Aug/2014:12|wc -l 0
    cat access_log|grep 27/Aug/2014:13|wc -l 49
    cat access_log|grep 27/Aug/2014:14|wc -l 82
    cat access_log|grep 27/Aug/2014:15|wc -l 0
    cat access_log|grep 27/Aug/2014:16|wc -l 384
    cat access_log|grep 27/Aug/2014:17|wc -l 918
    cat access_log|grep 27/Aug/2014:18|wc -l 368
    cat access_log|grep 27/Aug/2014:19|wc -l 184
    cat access_log|grep 27/Aug/2014:20|wc -l 897
    cat access_log|grep 27/Aug/2014:21|wc -l 202
    cat access_log|grep 27/Aug/2014:22|wc -l 116
    cat access_log|grep 27/Aug/2014:23|wc -l 159

    [root@cms02 logs]# DAY=28/Aug/2014 apachehits.py
    cat access_log|grep 28/Aug/2014:00|wc -l 187
    cat access_log|grep 28/Aug/2014:01|wc -l 361
    cat access_log|grep 28/Aug/2014:02|wc -l 356
    cat access_log|grep 28/Aug/2014:03|wc -l 273
    cat access_log|grep 28/Aug/2014:04|wc -l 388
    cat access_log|grep 28/Aug/2014:05|wc -l 402
    cat access_log|grep 28/Aug/2014:06|wc -l 338
    cat access_log|grep 28/Aug/2014:07|wc -l 339
    cat access_log|grep 28/Aug/2014:08|wc -l 667
    cat access_log|grep 28/Aug/2014:09|wc -l 407
    cat access_log|grep 28/Aug/2014:10|wc -l 1040
    cat access_log|grep 28/Aug/2014:11|wc -l 346
    cat access_log|grep 28/Aug/2014:12|wc -l 258
    cat access_log|grep 28/Aug/2014:13|wc -l 325
    cat access_log|grep 28/Aug/2014:14|wc -l 393
    cat access_log|grep 28/Aug/2014:15|wc -l 0
    cat access_log|grep 28/Aug/2014:16|wc -l 0
    cat access_log|grep 28/Aug/2014:17|wc -l 0
    cat access_log|grep 28/Aug/2014:18|wc -l 0
    cat access_log|grep 28/Aug/2014:19|wc -l 0
    cat access_log|grep 28/Aug/2014:20|wc -l 0
    cat access_log|grep 28/Aug/2014:21|wc -l 0
    cat access_log|grep 28/Aug/2014:22|wc -l 0
    cat access_log|grep 28/Aug/2014:23|wc -l 0
    [root@cms02 logs]# 






Aug 4, 2014
------------

Following a powercut cannot connect to cms02 over ssh, although 
looks normal at console login.

Stays same after commandline reboot.

::


    [blyth@cms01 ~]$ uptime
     11:41:18 up  3:31,  1 user,  load average: 0.00, 0.02, 0.00

    delta:~ blyth$ ping cms02.phys.ntu.edu.tw
    PING cms02.phys.ntu.edu.tw (140.112.101.191): 56 data bytes
    Request timeout for icmp_seq 0

    delta:~ blyth$ ping 140.112.101.191
    PING 140.112.101.191 (140.112.101.191): 56 data bytes
    Request timeout for icmp_seq 0

    [blyth@cms01 ~]$ ping 140.112.101.191
    PING 140.112.101.191 (140.112.101.191) 56(84) bytes of data.
    From 140.112.101.190 icmp_seq=1 Destination Host Unreachable
    From 140.112.101.190 icmp_seq=2 Destination Host Unreachable


Looks like some network infrastructure did not come back following 
power outage.




Following Typhoon Matmo C2R to H1 backups failing 
------------------------------------------------------

Possibly due to this ssh trouble::

    [root@cms02 ~]# ssh H1
    reverse mapping checking getaddrinfo for hep1.phys.ntu.edu.tw failed - POSSIBLE BREAKIN ATTEMPT!
    Enter passphrase for key '/root/.ssh/id_rsa': 
    Last login: Thu Jul 24 14:05:43 2014 from cms02.phys.ntu.edu.tw
    [blyth@hep1 ~]$ uptime
    14:07:36 up  4:08,  2 users,  load average: 0.00, 0.00, 0.00


Attack 19/Jun/2014 from 183.60.119.35
---------------------------------------

::

    delta:cms02 blyth$  LOG=Jun_2014_access_log DAY=19/Jun/2014 apachehits.py
    cat Jun_2014_access_log|grep 19/Jun/2014:00|wc -l 421
    cat Jun_2014_access_log|grep 19/Jun/2014:01|wc -l 383
    cat Jun_2014_access_log|grep 19/Jun/2014:02|wc -l 422
    cat Jun_2014_access_log|grep 19/Jun/2014:03|wc -l 355
    cat Jun_2014_access_log|grep 19/Jun/2014:04|wc -l 379
    cat Jun_2014_access_log|grep 19/Jun/2014:05|wc -l 2794
    cat Jun_2014_access_log|grep 19/Jun/2014:06|wc -l 4
    cat Jun_2014_access_log|grep 19/Jun/2014:07|wc -l 0
    cat Jun_2014_access_log|grep 19/Jun/2014:08|wc -l 0
    cat Jun_2014_access_log|grep 19/Jun/2014:09|wc -l 0
    cat Jun_2014_access_log|grep 19/Jun/2014:10|wc -l 0
    cat Jun_2014_access_log|grep 19/Jun/2014:11|wc -l 149
    cat Jun_2014_access_log|grep 19/Jun/2014:12|wc -l 129
    cat Jun_2014_access_log|grep 19/Jun/2014:13|wc -l 137
    cat Jun_2014_access_log|grep 19/Jun/2014:14|wc -l 200
    cat Jun_2014_access_log|grep 19/Jun/2014:15|wc -l 129
    cat Jun_2014_access_log|grep 19/Jun/2014:16|wc -l 99
    cat Jun_2014_access_log|grep 19/Jun/2014:17|wc -l 146
    cat Jun_2014_access_log|grep 19/Jun/2014:18|wc -l 114
    cat Jun_2014_access_log|grep 19/Jun/2014:19|wc -l 218
    cat Jun_2014_access_log|grep 19/Jun/2014:20|wc -l 346
    cat Jun_2014_access_log|grep 19/Jun/2014:21|wc -l 348
    cat Jun_2014_access_log|grep 19/Jun/2014:22|wc -l 327
    cat Jun_2014_access_log|grep 19/Jun/2014:23|wc -l 393

    grep 19/Jun/2014:05 Jun_2014_access_log | cut -d " " -f1 


::

    delta:cms02 blyth$ LOG=Jun_2014_access_log DAY=19/Jun/2014 IP=183.60.119.35 apachehits.py
    cat Jun_2014_access_log|grep ^183.60.119.35|grep 19/Jun/2014:00|wc -l 0
    cat Jun_2014_access_log|grep ^183.60.119.35|grep 19/Jun/2014:01|wc -l 0
    cat Jun_2014_access_log|grep ^183.60.119.35|grep 19/Jun/2014:02|wc -l 0
    cat Jun_2014_access_log|grep ^183.60.119.35|grep 19/Jun/2014:03|wc -l 0
    cat Jun_2014_access_log|grep ^183.60.119.35|grep 19/Jun/2014:04|wc -l 0
    cat Jun_2014_access_log|grep ^183.60.119.35|grep 19/Jun/2014:05|wc -l 2466
    cat Jun_2014_access_log|grep ^183.60.119.35|grep 19/Jun/2014:06|wc -l 0
    cat Jun_2014_access_log|grep ^183.60.119.35|grep 19/Jun/2014:07|wc -l 0
    cat Jun_2014_access_log|grep ^183.60.119.35|grep 19/Jun/2014:08|wc -l 0
    cat Jun_2014_access_log|grep ^183.60.119.35|grep 19/Jun/2014:09|wc -l 0
    cat Jun_2014_access_log|grep ^183.60.119.35|grep 19/Jun/2014:10|wc -l 0
    cat Jun_2014_access_log|grep ^183.60.119.35|grep 19/Jun/2014:11|wc -l 0
    cat Jun_2014_access_log|grep ^183.60.119.35|grep 19/Jun/2014:12|wc -l 0








Confirmed Robot Attack 20/Jun/2014 from 58.254.168.39
--------------------------------------------------------

::

    delta:cms02 blyth$ LOG=Jun_2014_access_log DAY=20/Jun/2014 apachehits.py
    grep 20/Jun/2014:00 Jun_2014_access_log | wc -l  423
    grep 20/Jun/2014:01 Jun_2014_access_log | wc -l  684
    grep 20/Jun/2014:02 Jun_2014_access_log | wc -l  620
    grep 20/Jun/2014:03 Jun_2014_access_log | wc -l  602
    grep 20/Jun/2014:04 Jun_2014_access_log | wc -l  627
    grep 20/Jun/2014:05 Jun_2014_access_log | wc -l  600
    grep 20/Jun/2014:06 Jun_2014_access_log | wc -l  623
    grep 20/Jun/2014:07 Jun_2014_access_log | wc -l  2781
    grep 20/Jun/2014:08 Jun_2014_access_log | wc -l  0
    grep 20/Jun/2014:09 Jun_2014_access_log | wc -l  0
    grep 20/Jun/2014:10 Jun_2014_access_log | wc -l  0
    grep 20/Jun/2014:11 Jun_2014_access_log | wc -l  2723
    grep 20/Jun/2014:12 Jun_2014_access_log | wc -l  0
    grep 20/Jun/2014:13 Jun_2014_access_log | wc -l  0
    grep 20/Jun/2014:14 Jun_2014_access_log | wc -l  0
    grep 20/Jun/2014:15 Jun_2014_access_log | wc -l  0


    delta:cms02 blyth$ LOG=Jun_2014_access_log DAY=20/Jun/2014 IP=58.254.168.39 apachehits.py
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:00|wc -l 0
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:01|wc -l 0
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:02|wc -l 0
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:03|wc -l 0
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:04|wc -l 0
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:05|wc -l 0
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:06|wc -l 0
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:07|wc -l 2440
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:08|wc -l 0
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:09|wc -l 0
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:10|wc -l 0
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:11|wc -l 2640
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:12|wc -l 0
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:13|wc -l 0
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:14|wc -l 0
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:15|wc -l 0
    cat Jun_2014_access_log|grep ^58.254.168.39|grep 20/Jun/2014:16|wc -l 0



Normal Hourly Hits
-------------------

::

    delta:cms02 blyth$ LOG=Jun_2014_access_log DAY=18/Jun/2014 apachehits.py
    cat Jun_2014_access_log|grep 18/Jun/2014:00|wc -l 512
    cat Jun_2014_access_log|grep 18/Jun/2014:01|wc -l 156
    cat Jun_2014_access_log|grep 18/Jun/2014:02|wc -l 215
    cat Jun_2014_access_log|grep 18/Jun/2014:03|wc -l 180
    cat Jun_2014_access_log|grep 18/Jun/2014:04|wc -l 252
    cat Jun_2014_access_log|grep 18/Jun/2014:05|wc -l 205
    cat Jun_2014_access_log|grep 18/Jun/2014:06|wc -l 333
    cat Jun_2014_access_log|grep 18/Jun/2014:07|wc -l 358
    cat Jun_2014_access_log|grep 18/Jun/2014:08|wc -l 296
    cat Jun_2014_access_log|grep 18/Jun/2014:09|wc -l 321
    cat Jun_2014_access_log|grep 18/Jun/2014:10|wc -l 299
    cat Jun_2014_access_log|grep 18/Jun/2014:11|wc -l 380
    cat Jun_2014_access_log|grep 18/Jun/2014:12|wc -l 482
    cat Jun_2014_access_log|grep 18/Jun/2014:13|wc -l 372
    cat Jun_2014_access_log|grep 18/Jun/2014:14|wc -l 408
    cat Jun_2014_access_log|grep 18/Jun/2014:15|wc -l 359
    cat Jun_2014_access_log|grep 18/Jun/2014:16|wc -l 348
    cat Jun_2014_access_log|grep 18/Jun/2014:17|wc -l 358
    cat Jun_2014_access_log|grep 18/Jun/2014:18|wc -l 317
    cat Jun_2014_access_log|grep 18/Jun/2014:19|wc -l 279
    cat Jun_2014_access_log|grep 18/Jun/2014:20|wc -l 321
    cat Jun_2014_access_log|grep 18/Jun/2014:21|wc -l 309
    cat Jun_2014_access_log|grep 18/Jun/2014:22|wc -l 184
    cat Jun_2014_access_log|grep 18/Jun/2014:23|wc -l 412





Jun 20, 2014 : again
---------------------

Another early morning fail and missed 07:42 check::

    curl -s --connect-timeout 3 http://dayabay.phys.ntu.edu.tw/repos/env/ 

    date                 val       
    -------------------  ----------
    2014-06-20T10:42:01  0.0       
    2014-06-20T09:42:01  0.0       
    2014-06-20T08:42:01  0.0       
    2014-06-20T06:42:01  1.0       
    2014-06-20T05:42:01  1.0       
    2014-06-20T04:42:04  1.0       
    2014-06-20T03:42:02  1.0       
    2014-06-20T02:42:01  1.0       


Working assumption is that rogue spiders are hitting on apache too much.  
Need to examine `access_log` to see if that is the case and block the offending IPs.  


#. After reboot, observe highly loaded machine and a screen full of httpd processes in top
#. It takes more than 5min for httpd to stop after `/sbin/service httpd stop`



Jun 19, 2014 : httpd offline, OOM again
-----------------------------------------

#. valmon monitoring indicates apache SVN fail  
#. no httpd, pingable but cannot SSH in 
#. ~11:00 reboot 

   * restores SSH access
   * but httpd does not come back automatically ? 


hourly valmon monitoring fails from 06:42 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



::

    curl -s --connect-timeout 3 http://dayabay.phys.ntu.edu.tw/repos/env/ 

    date                 val       
    -------------------  ----------
    2014-06-19T10:42:01  0.0       
    2014-06-19T09:42:01  0.0       
    2014-06-19T08:42:01  0.0       
    2014-06-19T07:42:01  0.0       
    2014-06-19T06:42:01  0.0       
    2014-06-19T05:42:01  1.0       
    2014-06-19T04:42:01  1.0       
    2014-06-19T03:42:01  1.0       
    2014-06-19T02:42:01  1.0       
    2014-06-19T01:42:01  1.0       






Original Cause, httpd OOM
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [root@cms02 log]# grep oom messages
    Jun 19 06:40:41 cms02 kernel: oom-killer: gfp_mask=0x1d2
    Jun 19 07:28:57 cms02 kernel: oom-killer: gfp_mask=0xd2
    Jun 19 08:04:39 cms02 kernel: oom-killer: gfp_mask=0xd0
    Jun 19 08:38:38 cms02 kernel: oom-killer: gfp_mask=0x1d2
    Jun 19 08:40:13 cms02 kernel: oom-killer: gfp_mask=0x1d2
    Jun 19 09:21:33 cms02 kernel: oom-killer: gfp_mask=0x1d2
    Jun 19 10:21:45 cms02 kernel: oom-killer: gfp_mask=0xd2
    Jun 19 10:27:25 cms02 kernel: oom-killer: gfp_mask=0xd2
    Jun 19 10:29:30 cms02 kernel: oom-killer: gfp_mask=0x1d2
    Jun 19 10:56:40 cms02 kernel: oom-killer: gfp_mask=0x1d2
    [root@cms02 log]# 


Restart httpd
~~~~~~~~~~~~~~~~

::

    [root@cms02 log]# /sbin/service httpd start


