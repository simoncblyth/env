#!/usr/bin/env python
"""
Top 30 hitters in lask 10k hits on cms02::

    [root@cms02 logs]# access.py 
    uip 578
    ip 10000
            66.249.77.98 : 1353        # google
            207.46.13.35 : 483         # microsoft
            207.46.13.97 : 457         # microsoft 
           157.55.39.113 : 448         # microsoft  
            66.249.77.93 : 396         # google
           157.55.39.210 : 385         # microsoft
           66.249.79.247 : 368         # google
           66.249.79.234 : 351 
           66.249.79.249 : 341 
         178.154.255.133 : 201         # yandex 
         178.154.255.142 : 200 
         178.154.255.129 : 197 
         178.154.255.131 : 193 
         178.154.255.132 : 192 
         178.154.255.135 : 192 
         178.154.255.130 : 190 
         178.154.255.143 : 189 
         178.154.255.140 : 189 
         178.154.255.139 : 188 
         178.154.255.134 : 177 
         178.154.255.141 : 171 
         178.154.255.138 : 171 
           66.249.79.241 : 154 
           66.249.79.228 : 150 
          140.112.102.77 : 134 
           66.249.79.252 : 114 
           5.255.253.193 : 112 
           207.46.13.130 : 57 
         140.112.101.190 : 54 
             159.93.14.8 : 43 

"""
import os

def access():
    ip_  = "head -10000 access_log | cut -d ' ' -f1 | sort "
    uip_ = ip_ + "| uniq "
    uip = os.popen(uip_).read().split()
    ip  = os.popen(ip_).read().split()

    print "uip", len(uip)
    print "ip", len(ip)
    d = dict([(_,ip.count(_)) for _ in ip])
    print "\n".join(["%20s : %s " % (k,d[k]) for k in sorted(d, reverse=True, key=lambda k:d[k])[:30]]) 


if __name__ == '__main__':
    access()



