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
    """
    http://g4pb.local/tracs/workflow/ticket/16

    Are missing attachments message/link
    """
    def __init__(self, db, id_):
        self.db = db
        self.id_ = id_ 
        self.name = id_
        recs = db("select keywords from ticket where id=%(id_)s ;" % locals() )
        assert len(recs) == 1
        tags = recs[0]["keywords"].split()

       


        tkt, tktTab = self.make_ticket_table(db, id_)
        self.tkt = tkt
        self.tkt_table = tktTab

        md = collections.OrderedDict()  
        md["name"] = id_
        md["time"] = tkt["time"] 
        md["ftime"] = tkt["ftime"] 
        md["tags"] = " ".join(tags)
 
        self.tags = tags      
        self.metadict = md      


        edits, editsTab = self.make_edits_table(db, id_)
        self.edits = edits
        self.edits_table = editsTab

        changes = self.make_changes(db, id_, edits)
        self.changes = changes

        pass

    def __repr__(self):
        return "<TracTicketPage %3d edits %d  changes %d> " % ( self.id_, len(self.edits), len(self.changes) )

    def __unicode__(self):
        """
        The tables are already rst so incorporate them after description and changes have been translated
        """
        return "\n".join(["= %s =" % self.title, "", self.description, "","" ] + map(unicode, self.changes))

    def __str__(self):
        return unicode(self).encode('utf-8')
 
    title = property(lambda self:"Ticket %(id)s : %(summary)s " % self.tkt )
    description = property(lambda self:self.tkt["description"].replace('\r\n','\n') )


    def make_ticket_table(self, db, id_):
        tkt_ = db("select * from ticket where id=%d" % id_ )
        assert len(tkt_) == 1
        tkt = tkt_[0]

        fields0 = "id type status resolution severity reporter owner ftime fchangetime priority resolution milestone component cc keywords version"
        fields1 = "description summary"

        tkt["ftime"] = ftime_(tkt["time"])
        tkt["fchangetime"] = ftime_(tkt["changetime"])

        tab = Table(hdr=True)
        cols = ["qty", "value"]
        tab.append(cols)
        for k in fields0.split():
            tab.append([k, unicode(tkt[k])])
        pass
        return tkt, tab
       

    def make_edits_table(self, db, id_):
        fields = map(lambda _:_.strip(), filter(lambda _:len(_.strip()),r"""
        time
        datetime(time, 'unixepoch', 'localtime') as timeLocal
        group_concat(author) as author
        group_concat(field) as fields
        """.split("\n")))

        fields = ",".join(fields)

        sql = "select %(fields)s from ticket_change where ticket='%(id_)s' group by time" % locals() 
        edits = db(sql)

        tab = Table(hdr=True)
        cols = ["time", "timeLocal", "author", "fields" ]
        tab.append(cols)

        for edit in edits:
            tab.append([unicode(edit[k]) for k in cols]) 
        pass
        return edits, tab


    def make_changes(self, db, id_, edits):
        changes = []
        for edit in edits:
            time = edit["time"]
            fields = edit["fields"]
            tc = TicketChange(time=time, fields=fields) 
            for field in fields.split(","):
                sql = "select oldvalue, newvalue from ticket_change where ticket='%(id_)s' and field='%(field)s' and time='%(time)s'" % locals()
                recs = db(sql)
                assert len(recs) == 1
                rec = recs[0]
                tc.append(FieldChange(field=field, oldvalue=rec['oldvalue'], newvalue=rec['newvalue'] ))
            pass
            changes.append(tc)
        pass
        return changes



class FieldChange(dict):
    def __init__(self, *args, **kwa):
        dict.__init__(self, *args, **kwa)
    def __unicode__(self):
        if self["field"] == "comment":
             ret = self["newvalue"].replace('\r\n','\n') 
        else:
             ret = " * %(field)s : %(oldvalue)s -> %(newvalue)s " % self
        pass
        return ret

class TicketChange(list):
    def __init__(self, *args, **kwa):
        time = kwa.pop("time")
        fields = kwa.pop("fields")
        list.__init__(self, *args, **kwa)
        self.time = time
        self.fields = fields
   
    def __unicode__(self):
        return "\n".join(["","","== TC %s : %s ==" % (ftime_(self.time), self.fields)] + [""] + map(unicode, self))
        



if __name__ == '__main__':

    level = 'INFO'
    logging.basicConfig(level=getattr(logging, level)) 
    from env.sqlite.db import DB
    from env.web.cnf import Cnf
 
    cnf = Cnf.read("workflow_trac2sphinx", "~/.workflow.cnf") 

    db = DB(cnf["tracdb"], asdict=True)
 
    tp = TracTicketPage(db,  int(sys.argv[1]) )
    print tp
        



