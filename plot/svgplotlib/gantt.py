#!/usr/bin/env python2.6
"""
  File "gantt.py", line 3, in <module>
      from svgplotlib import Gantt, Duration
        File "/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/svgplotlib/__init__.py", line 6, in <module>
            from collections import namedtuple
            ImportError: cannot import name namedtuple


namedtuple, New in version 2.6.

"""
from datetime import date
from svgplotlib import Gantt, Duration

items = []
items.append(Duration('Item 1', date(2009, 1, 4), date(2009, 8, 10), '90%'))
items.append(Duration('Item 2', date(2009, 3, 11), date(2009, 8, 17), '50%'))
items.append(Duration('Item 3', date(2009, 4, 18), date(2009, 8, 24), '70%'))
items.append(Duration('Item 4', date(2009, 5, 25), date(2009, 8, 31), '10%'))
items.append(Duration('Item 4', date(2009, 5, 25), date(2009, 9, 27), '25%'))

graph = Gantt(items)

