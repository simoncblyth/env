#!/usr/bin/env python
"""
tracticketpage.py 
==================

::

    ./tracticketpage.py 1

    ipython -i tracticketpage.py -- 1 



``ticket_change`` table has entries for every field that is changed, 
so it has lots of duplicated dates from each field changed by web edits. 

Best to express this with a group by time ? 

::

    sqlite> select ticket, time, datetime(time, 'unixepoch', 'localtime') as dtlocal, author, field, length(oldvalue), length(newvalue) from ticket_change ;
    ticket      time        dtlocal              author      field       length(oldvalue)  length(newvalue)
    ----------  ----------  -------------------  ----------  ----------  ----------------  ----------------
    1           1182101813  2007-06-18 01:36:53  blyth       status      3                 6               
    1           1182101813  2007-06-18 01:36:53  blyth       resolution  0                 5               
    1           1182101813  2007-06-18 01:36:53  blyth       comment     1                 13              
    2           1182406023  2007-06-21 14:07:03  blyth       comment     1                 296             
    3           1182529045  2007-06-23 00:17:25  blyth       comment     1                 35              
    3           1182592139  2007-06-23 17:48:59  blyth       status      3                 6               
    3           1182592139  2007-06-23 17:48:59  blyth       resolution  0                 5               
    3           1182592139  2007-06-23 17:48:59  blyth       comment     1                 36              
    6           1207260347  2008-04-04 06:05:47  blyth       comment     1                 367             
    6           1207260759  2008-04-04 06:12:39  blyth       status      3                 6               
    6           1207260759  2008-04-04 06:12:39  blyth       resolution  0                 5               
    6           1207260759  2008-04-04 06:12:39  blyth       comment     1                 7               
    9           1211784587  2008-05-26 14:49:47  blyth       descriptio  1440              1438            
    9           1211784587  2008-05-26 14:49:47  blyth       comment     1                 207             
    10          1211789455  2008-05-26 16:10:55  blyth       descriptio  13649             13647           
    10          1211789455  2008-05-26 16:10:55  blyth       comment     1                 160             
    12          1215414997  2008-07-07 15:16:37  blyth       status      3                 6               



    sqlite> select ticket, time, datetime(time, 'unixepoch', 'localtime') as dtl, group_concat(author) as author, group_concat(field) as fields from ticket_change group by time  ;
    tick  time             dtl                   author                          fields                        
    ----  ---------------  --------------------  ------------------------------  ------------------------------
    1     1182101813       2007-06-18 01:36:53   blyth,blyth,blyth               status,resolution,comment     
    2     1182406023       2007-06-21 14:07:03   blyth                           comment                       
    3     1182529045       2007-06-23 00:17:25   blyth                           comment                       
    3     1182592139       2007-06-23 17:48:59   blyth,blyth,blyth               status,resolution,comment     
    6     1207260347       2008-04-04 06:05:47   blyth                           comment                       
    6     1207260759       2008-04-04 06:12:39   blyth,blyth,blyth               status,resolution,comment     
    9     1211784587       2008-05-26 14:49:47   blyth,blyth                     description,comment           
    10    1211789455       2008-05-26 16:10:55   blyth,blyth                     description,comment           
    12    1215414997       2008-07-07 15:16:37   blyth,blyth,blyth               status,resolution,comment     
    13    1217336673       2008-07-29 21:04:33   blyth,blyth,blyth,blyth         status,resolution,description,
    14    1227971801       2008-11-29 23:16:41   blyth,blyth                     keywords,comment              
    15    1230112311       2008-12-24 17:51:51   blyth,blyth                     keywords,comment              
    16    1231344173       2009-01-08 00:02:53   blyth                           comment                       
    16    1231430680       2009-01-09 00:04:40   blyth                           comment                       
    16    1231432950       2009-01-09 00:42:30   blyth                           comment                       



"""
import logging, sys, re, os, collections, datetime
from env.trac.migration.tracwiki2rst import ListTagged
from env.doc.tabrst import Table
log = logging.getLogger(__name__)



ftime_ = lambda _:datetime.datetime.fromtimestamp(_).strftime('%Y-%m-%dT%H:%M:%S' )


class TracTicketPage(object):
    def __init__(self, db, id_):
        tkt_ = db("select * from ticket where id=%d" % id_ )
        assert len(tkt_) == 1
        tkt = tkt_[0]

        chgs = db("select * from ticket_change where ticket=%d" % id_ )

        self.id_ = id_ 
        self.tkt = tkt
        self.chgs = chgs

        fields0 = "id type status resolution severity reporter owner ftime fchangetime priority resolution milestone component cc keywords version"
        fields1 = "description summary"

        tkt["ftime"] = ftime_(tkt["time"])
        tkt["fchangetime"] = ftime_(tkt["changetime"])

        tab = Table(hdr=True)
        cols = ["qty", "value"]
        tab.append(cols)
        for k in fields0.split():
            tab.append([k, unicode(self.tkt[k])])
        pass
        self.tab = tab 
        self.title = "Ticket %(id)s : %(summary)s " % tkt



    def __repr__(self):
        return "<TracTicketPage %3d chgs %d > " % ( self.id_, len(self.chgs) )


if __name__ == '__main__':

    level = 'INFO'
    logging.basicConfig(level=getattr(logging, level)) 
    from env.sqlite.db import DB
    from env.web.cnf import Cnf
 
    cnf = Cnf.read("workflow_trac2sphinx", "~/.workflow.cnf") 

    db = DB(cnf["tracdb"], asdict=True)
 
    tktid = int(sys.argv[1])
    tp = TracTicketPage(db, tktid)

    print tp
    print tp.title
    print tp.tab
    



