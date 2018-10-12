#!/usr/bin/env python
"""

::

    simon:env blyth$ calendar.py 6
    Wed 01 
    Thu 02 
    Fri 03 
    Sat 04 
    Sun 05 

    Mon 06 
    Tue 07 
    Wed 08 
    Thu 09 
    Fri 10 
    Sat 11 


"""
import sys
days = "Mon Tue Wed Thu Fri Sat Sun".split()

def calendar(monday):
    for i in range(1,31):
        day = (i-monday) % 7 
        print("%s %0.2d " % ( days[day], i ))
        if day == 6:print()
    pass


if __name__ == '__main__':
    monday = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    calendar(monday)

    



