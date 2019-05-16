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

#FMT = "%c"  ## Tue Apr 30 13:39:30 2019
FMT = "%F"   ##  '2019-04-30'
fmt_ = lambda dt:dt.strftime(FMT)

class Day(object):
    def __init__(self, text, default=None):
        text = text.replace("_", " ")
        self.text = text 
        self.dt = parse(text, default=default)      
    def __repr__(self):
        return "<Day %10s : %s>" % ( self.text, fmt_(self.dt)) 

class Days(list):
    """
    Simply a list of Day instances
    """
    def __init__(self, *args, **kwa):
        list.__init__(self, *args, **kwa)

    def periods(self):
        """
        The sequence is assumed to be a complete ordered 
        record of the dates of arrivals and departures 
        from a single location/country.

        To handle edge periods at the beginning and end of 
        a year bracket the entries and exits with
        artifical Jan1 entry/exit dates::

            days.py Jan1_2019 real-dates-here Jan1_2020 

        This has the advantage that within+without day counts
        should match the total number of days in the year: 365/364.

        The periods between the times are calculated and 
        the even and odd periods are summed to provide the 
        total number of days within and without the location.
        Whether even/odd is within/without depends on the meaning
        given to the sequence of dates provided. 

        NB for reproducibility specify the year, 
        otherwise dates will defaut to the current year making 
        results depend on when run.
        """
        tot = [0,0,0] 
        lines = []
        for i in range(1,len(self)):
            dt0 = self[i-1].dt
            dt1 = self[i].dt
            dt = dt1 - dt0 
            dtd = dt.days

            tot[i % 2] += dtd  ##  1 for odd i,  0 for even i  
            tot[2] += dtd      ##  sum of within and without 

            per = " %2d : %12s %12s : %10s " % (  i, fmt_(dt0), fmt_(dt1), dtd ) 
            line =  "%s : %s " % (per, tot )    
            lines.append(line)
        pass
        return lines

    def __repr__(self):
        return "\n".join(map(repr, self))
    def __str__(self):
        return "\n".join( map(repr,self) + [""] + self.periods() )



if __name__ == '__main__':


    daylist = []
    args = sys.argv[1:]
    last = None
    for arg in args:
        day = Day(arg, default=last)
        last = day.dt
        daylist.append(day)
    pass
    days = Days(daylist)
    print("\n".join(days.periods()))  


