cms01 DNS HTTP 10 seconds delay
==================================

.. contents:: :local:


cms01 access monitoring is failing 
-------------------------------------

My hourly access/speed test on cms01 began failing about 24hrs ago, sending mails like the below every hour:: 

    cms01.phys.ntu.edu.tw : valmon.py -s envmon : 2013-11-20T09:42:02 :    val == 1 : False 

      val == 1                                 : False 

      summary                                  : all:1 True:0 False:1  
      context                                  : {'val': 0.0} 


    % ~/.env.cnf blyth@cms01.phys.ntu.edu.tw 
    [envmon]
    tn = envmon 
    instruction = require a single trunk to be found, verifying that the apache interface to SVN is working 
    return = int 
    dbpath = ~/.env/envmon.sqlite 
    cmd = curl -s --connect-timeout 3 http://dayabay.phys.ntu.edu.tw/repos/env/ | grep trunk | wc -l 
    note = check C2 server from cron on other nodes 
    email = blyth@hep1.phys.ntu.edu.tw 
    constraints = ( val == 1, ) 

    date                 val       
    -------------------  ----------
    2013-11-20T09:42:02  0.0       
    2013-11-20T08:42:01  0.0       
    2013-11-20T07:42:01  0.0       
    2013-11-20T06:42:01  0.0       
    2013-11-20T05:42:01  0.0       
    2013-11-20T04:42:01  0.0       
    2013-11-20T03:42:01  0.0       
    2013-11-20T02:42:02  0.0       
    2013-11-20T01:42:01  0.0       
    2013-11-20T00:42:01  0.0       
    2013-11-19T23:42:01  0.0       
    2013-11-19T22:42:01  0.0       
    2013-11-19T21:42:01  0.0       
    2013-11-19T20:42:01  0.0       
    2013-11-19T19:42:01  0.0       
    2013-11-19T18:42:01  0.0       
    2013-11-19T17:42:03  0.0       
    2013-11-19T16:42:02  0.0       
    2013-11-19T15:42:01  0.0       
    2013-11-19T14:42:01  0.0       
    2013-11-19T13:42:02  0.0       
    2013-11-19T12:42:01  0.0       
    2013-11-19T11:42:01  0.0       
    2013-11-19T10:42:01  1.0       

Avoiding DNS lookup avoids the delay::

    [blyth@cms01 ~]$ time curl http://140.112.101.191/repos/env/ | grep trunk
      <li><a href="trunk/">trunk/</a></li>
    real    0m0.011s
    user    0m0.005s
    sys     0m0.002s

A reproducible DNS delay of 10 seconds for http GET from cms01 is apparent::

    [blyth@cms01 ~]$ time curl  http://dayabay.phys.ntu.edu.tw/repos/env/ | grep trunk
      <li><a href="trunk/">trunk/</a></li>
    real    0m10.012s
    user    0m0.005s
    sys     0m0.004s

    [blyth@cms01 ~]$ time curl  http://dayabay.phys.ntu.edu.tw/repos/env/  | grep trunk
      <li><a href="trunk/">trunk/</a></li>
    real    0m10.012s
    user    0m0.005s
    sys     0m0.004s
    [blyth@cms01 ~]$ 

Not just from cms02::

    [blyth@cms01 ~]$ time curl http://www.google.com
    <HTML><HEAD><meta http-equiv="content-type" content="text/html;charset=utf-8">
    <TITLE>302 Moved</TITLE></HEAD><BODY>
    <H1>302 Moved</H1>
    The document has moved
    <A HREF="http://www.google.com.tw/?gws_rd=cr&amp;ei=DCuMUu-rL62RiQe4yYGIBA">here</A>.
    </BODY></HTML>

    real    0m10.125s
    user    0m0.005s
    sys     0m0.003s
    [blyth@cms01 ~]$ 


John sent a mail about DNS recently::

    One old DNS server 140.112.101.1 was decommissioned by our dept. system
    manager, Debbie. If you are still using this DNS in your machine, you will
    encounter some problems. Please change to 140.112.254.4 or 140.112.2.2 or other
    DNS servers, ex 168.95.1.1 or 8.8.8.8.


Other Observations
------------------

#. ssh logins to cms01,cms02,hfag have been slow for the past day, several minutes in some cases

DNS /etc/resolv.conf
-----------------------

How to change DNS ?

* :google:`linux change DNS server`
* http://www.rackspace.com/knowledge_center/article/changing-dns-settings-on-linux
* add function ``dns-edit`` to do this


::

    [blyth@cms01 ~]$ cat /etc/resolv.conf
    search heplocal
    search phys.ntu.edu.tw
    nameserver 140.112.101.1
    nameserver 140.112.2.2
    nameserver 168.95.1.1

::

    [blyth@cms02 ~]$ cat /etc/resolv.conf
    nameserver 140.112.101.1

    [blyth@cms02 ~]$ curl http://www.google.com
    curl: (6) Couldn't resolve host 'www.google.com'

::

    [blyth@hfag blyth]$ cat /etc/resolv.conf
    nameserver 140.112.101.1
    search phys.ntu.edu.tw


Fixed with ``dns-edit``
-------------------------

#. also note that SSH logins are back to normal, that must be doing some DNS lookups

::

    [blyth@cms01 e]$ dns-
    [blyth@cms01 e]$ dns-edit
    === dns-edit : sudo vi /etc/resolv.conf
    Password:
    [blyth@cms01 e]$ 
    [blyth@cms01 e]$ 
    [blyth@cms01 e]$ time curl -s http://www.google.com
    <HTML><HEAD><meta http-equiv="content-type" content="text/html;charset=utf-8">
    <TITLE>302 Moved</TITLE></HEAD><BODY>
    <H1>302 Moved</H1>
    The document has moved
    <A HREF="http://www.google.com.tw/?gws_rd=cr&amp;ei=-y-MUoPPN-6ZiAfH3oDQBQ">here</A>.
    </BODY></HTML>

    real    0m0.051s
    user    0m0.003s
    sys     0m0.006s
    [blyth@cms01 e]$ 



     
