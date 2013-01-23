#!/usr/bin/env python
"""
Obtains lists of tags for every Trac wikipage  

Usage::

   tractagsdump.py workflow_trac

Requirements
~~~~~~~~~~~~~

A Trac SQL report in slot 12 such as::

    select Name,Tag from tags where Tagspace = 'wiki'    

Configure to avoid pagination, so can get all tags at once::

    [report]
    items_per_page = 10000

TODO:

#. tags apply to tickets too, this script should handle that also 
#. how to persist the tags (maybe json sidecar files)

alternate
~~~~~~~~~~~~

Without controlling pagination would require per-page query::

   select Tag from tags where Tagspace = 'wiki'  and Name = '$NAME'   
   -- http://localhost/tracs/workflow/report/12?format=csv&NAME=PageName&USER=blyth

"""
from StringIO import StringIO
import csv
from pprint import pformat 
from env.web.cnf import cnf_
from env.web.autobrowser import AutoBrowser

def tags_by_page( browser, tags_url ):
    """
    :param browser: authenticated `AutoBrowser` instance
    :param tags_url: URL of a trac report that returns CSV of page names and tags

    Example of trac report SQL::

        select Name,Tag from tags where Tagspace = 'wiki'    

    And URL: http://localhost/tracs/workflow/report/12?format=csv
    """
    csvdata = browser.open_(tags_url) 
    csvfile = StringIO(csvdata)
    csvrdr = csv.reader(csvfile,delimiter=',')
    tags = {}
    for i,row in enumerate(csvrdr):
        if i==0:
            assert row == ['Name','Tag'], "unexpected 1st row %s " % row
        else:
            name,tag = row
            if name not in tags:
                tags[name] = []
            tags[name].append(tag)    
        pass
    return tags 


if __name__ == '__main__':
    cnf = cnf_(__doc__)
    browser = AutoBrowser(cnf)
    tags = tags_by_page( browser, cnf['tags_url'] ) 
    print pformat(tags)




    

