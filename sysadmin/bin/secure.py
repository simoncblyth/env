#!/usr/bin/env python
"""
Parse the SSHD logfile 

Jun 20 15:26:18 cms02 sshd[4395]: Accepted publickey for blyth from ::ffff:140.112.102.77 port 54420 ssh2
Jun 20 15:27:04 cms02 sudo:    blyth : TTY=pts/1 ; PWD=/home/blyth ; USER=root ; COMMAND=/sbin/chkconfig --list
Jun 20 15:27:28 cms02 sudo:    blyth : TTY=pts/1 ; PWD=/home/blyth ; USER=root ; COMMAND=/sbin/chkconfig --list httpd


"""
import os, re
from datetime import datetime
import time

# faux datetime.strptime that works in earlier pythons
strptime = lambda st,fmt:datetime(*(time.strptime(st, fmt)[0:6]))

now = datetime.now()

class Line(object):
    dt_len = 15 
    dt_fmt = "%b %d %H:%M:%S"
    node_ptn = re.compile("^\s*(\S*)\s*")
    pay_ptn = dict( 
            lmrx=re.compile("^\s*last message repeated (\d*) times"),
            sshd=re.compile("^\s*sshd\[\d*\]:(.*)$"),
            sudo= re.compile("^\s*sudo:(.*)$"),
    )

    sshd_ptn = (
       ("acckey",re.compile("^\s*Accepted publickey for (\S*) from ::ffff:(\S*) port (\d*) ssh2")),
       ("accpas",re.compile("^\s*Accepted password for (\S*) from ::ffff:(\S*) port (\d*) ssh2")), 
       ("discon",re.compile("^\s*Received disconnect from ::ffff:(\S*): (\d*): disconnected by user")),
       ("termin",re.compile("^\s*Received signal (\d*); terminating.")), 
       ("listen",re.compile("^\s*Server listening on :: port (\d*).")),
       ("debug",re.compile("^\s*debug1:(.*)")),
       ("other",re.compile("^\s*(.*)")),
    ) 

    def __init__(self, line):

        pay = self.parse_hdr(line)
        mpay = self.match( pay, self.pay_ptn.items() )  
        assert len(mpay) == 1 , "only one pay_ptn must match "
        payn,paym = mpay[0]
        payl = paym.groups()[0]

        self.mtch = payn              # lmrx/sshd/sudo   broad classification
        self.payl = payl

        sptn = getattr(self, "%s_ptn" % self.mtch , {} )
        if sptn:
            spay = self.match( payl, sptn )
            assert len(spay) > 0 , "at least one sub ptn must match "
            self.subm = spay[0][0]    
            self.subg = repr(spay[0][1].groups())    
        else:
            self.subm = "-"
            self.subg = "-"



    def parse_hdr(self, line ):
        """
        :param line: to be parsed

        Extract the date and node and return the remaining body of the line
        """
        dtt = line[0:self.dt_len]
        dts = line[self.dt_len:]
        self.dt = strptime("%s %s" % (now.year,dtt),"%Y " + self.dt_fmt)   # fudge the year, as not in the log
        m = self.node_ptn.match(dts)
        assert m, "failed to node match %s " % dts
        self.node = m.group(0)
        nend = m.span()[1]
        pay = dts[nend:].rstrip()
        return pay 


    def match( self, txt, mseq ):
        """
        :param txt: to be matched
        :param mseq: iterable providing key names and regexp patterns 
        """
        d = [] 
        for npt,ptn  in mseq:
            m = ptn.match(txt)
            if m:
               d.append((npt,m))
        return d  


    def __repr__(self):
        return "%s | %s | %s | %s | %s | %s " % ( self.dt.strftime("%c"), self.node, self.mtch, self.payl , self.subm, self.subg  )



class Secure(list):
    def __init__(self, paths=["/var/log/secure","/var/log/secure.1"] ):
        for path in paths:
            self.read(path)

    def read(self, path):
        fp = open(path,"r")
        for line_ in fp.readlines():
            line = Line(line_)
            self.append(line)


if __name__ == '__main__':
    s = Secure()
    s.sort(lambda a,b:cmp(a.dt,b.dt))   # no sorted in 2.3
    for i in s:
        #if i.mtch in ('sudo','lmrx'):
        #    pass
        #else:

        if i.mtch == 'sshd' and i.subm == 'other':
            print i



