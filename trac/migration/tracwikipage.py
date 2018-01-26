#!/usr/bin/env python
"""
tracwikipage.py 
================

::

    ./tracwikipage.py 3D
    ./tracwikipage.py MailDebug



"""
import logging, sys, re, os, collections, datetime
from env.trac.migration.tracwiki2rst import ListTagged
from env.doc.tabrst import Table
log = logging.getLogger(__name__)

class TracWikiPage(object):
    def __init__(self, db, name):

        tags = map(lambda _:str(_["tag"]), db("select tag from tags where tagspace=\"wiki\" and name=\"%s\" " % name ))
        rec = db("SELECT version,time,author,text,comment,readonly FROM wiki WHERE name=\"%s\" ORDER BY version DESC LIMIT 1" % name ) 
        assert len(rec) == 1 
        d = rec[0]

        assert type(d["author"]) is unicode
        assert type(d["text"]) is unicode
        assert type(d["comment"]) in (unicode, type(None))

        ftime = datetime.datetime.fromtimestamp(d["time"]).strftime('%Y-%m-%dT%H:%M:%S' )

        md = collections.OrderedDict()  
        md["name"] = name
        md["version"] = d["version"]
        md["time"] = d["time"]
        md["ftime"] = ftime
        md["author"] = d["author"]
        md["comment"] = d["comment"] if d["comment"] is not None else ""
        md["tags"] = " ".join(tags)

        self.db = db 
        self.name = name
        self.version = d["version"]
        self.time = d["time"]
        self.ftime = ftime
        self.author = d["author"]
        self.text = d["text"].replace('\r\n','\n')
        self.comment = d["comment"]
        self.tags = tags 
        self.metadict = md      
 
        full_table = self.make_history_table(name)
        smry_table = self.make_summary_table(name)
        pick_smry = len(full_table) > 20  
        history_table = smry_table if pick_smry else full_table

        if pick_smry:
            log.info(" %s : pick smry table :  full_table:%s smry_table:%s history_table:%s " % (name, len(full_table), len(smry_table), len(history_table) ))
        pass

        self.history_table = history_table
        self.full_table = full_table
        self.smry_table = smry_table


    def complete_ListTagged(self, lti, skipdoc=[]):
        """
        This appends the list of documents to the lti ListTagged instance

        http://www.sphinx-doc.org/en/stable/markup/inline.html#cross-referencing-documents
        """
        assert type(lti) is ListTagged, type(lti)
    
        skips = r"""
        or
        operator=union
        operation=union
        action=union
        """.lstrip().rstrip().split("\n")

        targ = lti.tags.lstrip().rstrip().replace(","," ")
        tags = filter(lambda _:not _ in skips, targ.split())

        stags = ",".join(map(lambda _:"'%s'" % _, tags ))
        sql = "select distinct name from tags where tagspace=\"wiki\" and tag in (%s) order by name ;" % stags 
        rec = self.db(sql)

        wikitagged = map(lambda _:_["name"], rec )
        for nm in wikitagged: 
            if nm in skipdoc:continue

            psql = "select tag from tags where name = \"%s\" order by tag ;" % nm  
            prec = self.db(psql)
            prec = map(lambda _:_['tag'], prec )
            lti.append("%s :doc:`%s` " % (nm,nm) )
        pass


    def make_history_table(self, name):
        """
        sqlite> select version, time, datetime(time, 'unixepoch', 'localtime') as dtlocal, author, comment from wiki where name='3D' ;
        version     time        dtlocal              author      comment   
        ----------  ----------  -------------------  ----------  ----------
        1           1234927211  2009-02-18 11:20:11  blyth                 
        2           1234927973  2009-02-18 11:32:53  blyth                 
        3           1234935187  2009-02-18 13:33:07  blyth                 
        4           1237285396  2009-03-17 18:23:16  blyth   
        """
        tab = Table(hdr=True)
        cols = ["version", "time", "dtlocal", "author", "comment"]
        tab.append(cols)
        sql = "select version, time, datetime(time, 'unixepoch', 'localtime') as dtlocal, author, comment from wiki where name='%s' ;" % name
        for d in self.db(sql):
            row = [unicode(d[c]) for c in cols]
            tab.append(row)
        pass
        return tab

    def make_summary_table(self, name):
        """
        Hmm need to suppress intraday debugging edits 

        sqlite> select name, version, comment from wiki where comment != "" and version > 5 ;
        name                            version     comment               
        ------------------------------  ----------  ----------------------
        WorkflowNav                     296         trial edit checking QC
        WorkflowNav                     297         remove belle          
        WorkflowNav                     299         NAS admin URLs        
        SCMBackup                       7           crontab rejig         
        Taipei                          9           transport and fora lin
        TravelCheckList                 12          move to NV Dropbox    
        BackupSetup                     39          zap non-ascii with {{{
        BackupSetup                     40          another couple of ille
        DiskSetup                       14          zap some escapes with 
        FlightPlanning                  6           Work Calendar : search
        sqlite> 


        sqlite> select min(version), max(version), min(time), max(time), datetime(time, 'unixepoch', 'localtime') as dtlocal, strftime("%Y-%m-%d", datetime(time, 'unixepoch', 'localtime')) as day from wiki where name="BackupSetup" group by day ;
        min(version)                    max(version)  min(time)         max(time)         dtlocal              day       
        ------------------------------  ------------  ----------------  ----------------  -------------------  ----------
        1                               17            1185678313.97008  1185723477.65012  2007-07-29 23:37:57  2007-07-29
        18                              19            1185725244.06869  1185790141.66226  2007-07-30 18:09:01  2007-07-30
        20                              20            1186302346.71912  1186302346.71912  2007-08-05 16:25:46  2007-08-05
        21                              22            1186370410.13802  1186370480.07885  2007-08-06 11:21:20  2007-08-06
        23                              23            1188970935.56765  1188970935.56765  2007-09-05 13:42:15  2007-09-05
        24                              25            1192596501.20154  1192614768.90187  2007-10-17 17:52:48  2007-10-17
        26                              29            1192688487.96013  1192688578.71093  2007-10-18 14:22:58  2007-10-18
        30                              33            1204360886        1204374219        2008-03-01 20:23:39  2008-03-01
        34                              34            1204446788        1204446788        2008-03-02 16:33:08  2008-03-02
        35                              36            1204534882        1204540625        2008-03-03 18:37:05  2008-03-03
        37                              38            1223905274        1223905305        2008-10-13 21:41:45  2008-10-13
        39                              40            1341833056        1341833263        2012-07-09 19:27:43  2012-07-09

        """

        fields = map(lambda _:_.strip(), filter(lambda _:len(_.strip()),r"""
        min(version) as minVer
        max(version) as maxVer
        min(time) as minTime
        max(time) as maxTime
        strftime("%Y-%m-%d", datetime(time, 'unixepoch', 'localtime')) as day
        group_concat(author, ",") as author
        group_concat(comment, " , ") as comment
        """.split("\n")))

        fields = ",".join(fields)

        sql = "select %(fields)s from wiki where name='%(name)s' group by day" % locals() 
        #print sql 

        tab = Table(hdr=True)
        cols = ["minVer", "maxVer", "day", "fminT", "fmaxT", "comment", "author"]
        tab.append(cols)

        for d in self.db(sql):
            minT = datetime.datetime.fromtimestamp(d["minTime"])
            maxT = datetime.datetime.fromtimestamp(d["maxTime"])
            d["fminT"] = minT.strftime('%H:%M:%S' )
            d["fmaxT"] = maxT.strftime('%H:%M:%S' )

            if d["comment"] is None:
                d["comment"] = ""
            elif len(d["comment"].replace(",", " ").strip()) == 0:
                d["comment"] = ""
            pass
            d["author"] = ",".join(list(set(d["author"].split(","))))
            tab.append([unicode(d[c]) for c in cols])
        pass
        return tab

    def __repr__(self):
        return "%5s : %30s : %10s : %15s : %60s : %s " % ( self.version, self.name, self.author, self.time, ",".join(self.tags), self.comment )

    def __unicode__(self):
        return "\n\n".join( [repr(self), unicode(self.text) ] ) 

    def __str__(self):
        return unicode(self).encode('utf-8')


if __name__ == '__main__':

    level = 'INFO'
    logging.basicConfig(level=getattr(logging, level)) 
    from env.sqlite.db import DB
    from env.web.cnf import Cnf
 
    cnf = Cnf.read("workflow_trac2sphinx", "~/.workflow.cnf") 
    db = DB(cnf["tracdb"], asdict=True)
 
    pgname = sys.argv[1] 
    wp = TracWikiPage(db, pgname)

    print wp
    print wp.full_table
    print wp.smry_table




