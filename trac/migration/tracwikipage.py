#!/usr/bin/env python
"""
tracwikipage.py 
================

::

    ./tracwikipage.py $(wtracdb-path) 3D
    ./tracwikipage.py $(wtracdb-path) MailDebug

"""
import logging, sys, re, os, collections, datetime
from env.trac.migration.tracwiki2rst import ListTagged
log = logging.getLogger(__name__)


class TracWikiPage(object):
    def __init__(self, db, name):

        tags = map(lambda _:str(_[0]), db("select tag from tags where tagspace=\"wiki\" and name=\"%s\" " % name ))
        rec = db("SELECT version,time,author,text,comment,readonly FROM wiki WHERE name=\"%s\" ORDER BY version DESC LIMIT 1" % name ) 
        version,time,author,text,comment,readonly = rec[0] 

        assert type(author) is unicode
        assert type(text) is unicode
        if comment is not None:
            assert type(comment) is unicode
        pass
        ftime = datetime.datetime.fromtimestamp(time).strftime('%Y-%m-%dT%H:%M:%S' )

        md = collections.OrderedDict()  
        md["name"] = name
        md["version"] = version
        md["time"] = time
        md["ftime"] = ftime
        md["author"] = author
        md["comment"] = comment if comment is not None else ""
        md["tags"] = " ".join(tags)

        self.db = db 

        self.name = name
        self.version = version
        self.time = time
        self.ftime = ftime
        self.author = author
        self.text = text.replace('\r\n','\n')
        self.comment = comment
        self.tags = tags 

        self.metadict = md      

    def complete_ListTagged(self, tgls):
        """
        http://www.sphinx-doc.org/en/stable/markup/inline.html#cross-referencing-documents
        """
        assert type(tgls) is ListTagged, type(tgls)
    
        skips = r"""
        or
        operator=union
        operation=union
        action=union
        """.lstrip().rstrip().split("\n")

        targ = tgls.tags.lstrip().rstrip().replace(","," ")
        tags = filter(lambda _:not _ in skips, targ.split())

        stags = ",".join(map(lambda _:"'%s'" % _, tags ))
        sql = "select distinct name from tags where tagspace=\"wiki\" and tag in (%s) order by name ;" % stags 
        rec = self.db(sql)

        wikitagged = map(lambda _:_[0], rec )
        for nm in wikitagged: 

            psql = "select tag from tags where name = \"%s\" order by tag ;" % nm  
            prec = self.db(psql)
            prec = map(lambda _:_[0], prec )
            ## hmm even when generate taglist only pages, still need to distingish 

            #prst = " ".join(["("] + map(lambda _:":doc:`%s <%s>`" % (_,_), prec ) + [")"])
            #prst2 = " ".join(["("] + map(lambda _:":ref:`%s`" % _, prec ) + [")"])
            #prst3 = " ".join(["("] + map(lambda _:":%s_" % _, prec ) + [")"])

            prst = ""
            prst2 = ""
            prst3 = ""


            tgls.append("%s :doc:`%s` %s %s %s" % (nm,nm, prst, prst2, prst3) )
        pass
        """  
        select distinct tag as t from tags order by tag ;
        select distinct tag as t from tags where t not in ( select distinct name from wiki ) order by tag ;
        select name, count(tag) as n from tags group by tag order by n desc ;
        """

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

    dbpath = sys.argv[1]
    pgname = sys.argv[2] 

    db = DB(dbpath)
    print db 
 
    wp = TracWikiPage(db, pgname)

    print wp



