#!/usr/bin/env python
"""
days.py
========

Calulate even/odd time period day totals from a list of in/out dates.

::

    delta:~ blyth$ ~/env/bin/days.py Jan_1 Jan_1_2019
    <Day      Jan 1 : Mon Jan  1 00:00:00 2018>
    <Day Jan 1 2019 : Tue Jan  1 00:00:00 2019>

      1 :       Mon Jan  1 00:00:00 2018       Tue Jan  1 00:00:00 2019 :        365  : [0, 365, 365] 
    delta:~ blyth$ 


    delta:~ blyth$ ~/env/bin/days.py Jan_1 Jul_3 Jan_1_2019
    <Day      Jan 1 : Mon Jan  1 00:00:00 2018>
    <Day      Jul 3 : Tue Jul  3 00:00:00 2018>
    <Day Jan 1 2019 : Tue Jan  1 00:00:00 2019>

      1 :       Mon Jan  1 00:00:00 2018       Tue Jul  3 00:00:00 2018 :        183  : [0, 183, 183] 
      2 :       Tue Jul  3 00:00:00 2018       Tue Jan  1 00:00:00 2019 :        182  : [182, 183, 365] 
    delta:~ blyth$ 


    delta:~ blyth$ ~/env/bin/days.py Jan_1 Jan_6 Feb_5 Feb_28 Jun_4 Oct_31 Jan_1_2019
    <Day      Jan 1 : Mon Jan  1 00:00:00 2018>
    <Day      Jan 6 : Sat Jan  6 00:00:00 2018>
    <Day      Feb 5 : Mon Feb  5 00:00:00 2018>
    <Day     Feb 28 : Wed Feb 28 00:00:00 2018>
    <Day      Jun 4 : Mon Jun  4 00:00:00 2018>
    <Day     Oct 31 : Wed Oct 31 00:00:00 2018>
    <Day Jan 1 2019 : Tue Jan  1 00:00:00 2019>

      1 :       Mon Jan  1 00:00:00 2018       Sat Jan  6 00:00:00 2018 :          5  : [0, 5, 5] 
      2 :       Sat Jan  6 00:00:00 2018       Mon Feb  5 00:00:00 2018 :         30  : [30, 5, 35] 
      3 :       Mon Feb  5 00:00:00 2018       Wed Feb 28 00:00:00 2018 :         23  : [30, 28, 58] 
      4 :       Wed Feb 28 00:00:00 2018       Mon Jun  4 00:00:00 2018 :         96  : [126, 28, 154] 
      5 :       Mon Jun  4 00:00:00 2018       Wed Oct 31 00:00:00 2018 :        149  : [126, 177, 303] 
      6 :       Wed Oct 31 00:00:00 2018       Tue Jan  1 00:00:00 2019 :         62  : [188, 177, 365] 
    delta:~ blyth$ 


"""
import sys
from dateutil.parser import parse
from datetime import datetime

fmt_ = lambda dt:dt.strftime("%c")

class Day(object):
    def __init__(self, text):
        text = text.replace("_", " ")
        self.text = text 
        self.dt = parse(text)      
    def __repr__(self):
        return "<Day %10s : %s>" % ( self.text, fmt_(self.dt)) 

class Days(list):
    def __init__(self, *args, **kwa):
        list.__init__(self, *args, **kwa)

    def periods(self):
        tot = [0,0,0] 
        lines = []
        for i in range(1,len(self)):
            dt0 = self[i-1].dt
            dt1 = self[i].dt
            dt = dt1 - dt0 
            dtd = dt.days

            tot[i % 2] += dtd 
            tot[2] += dtd

            per = " %2d : %30s %30s : %10s " % (  i, fmt_(dt0), fmt_(dt1), dtd ) 
            line =  "%s : %s " % (per, tot )    
            lines.append(line)
        pass
        return lines

    def __repr__(self):
        return "\n".join(map(repr, self))
    def __str__(self):
        return "\n".join( map(repr,self) + [""] + self.periods() )



if __name__ == '__main__':
    days = Days(map(Day, sys.argv[1:]))
    print days  


