#!/usr/bin/env python
"""
tracattachment.py 
==================

::

    sqlite> select * from attachment ;
    type        id                              filename                        size        time          description                              author               ipnr               
    ----------  ------------------------------  ------------------------------  ----------  ------------  ---------------------------------------  -------------------  -------------------
    wiki        AppScriptSafari                 favicon.png                     5832        1187343010    test attachment description              admin                ::1                
    wiki        Trac2LatexGrabbed               example-1.png                   1219        1188972625                                                                                     
    wiki        Trac2LatexGrabbed               example-1.pdf                   4496        1188972625                                                                                     
    wiki        Trac2LatexTesting               fig_math.png                    23920       1188972804                                                                                     
    wiki        Trac2LatexTesting               fig_math.pdf                    5122        1188972804                                                                                     
    wiki        FX                              SafariScreenSnapz001.png        42285       1196932034                                             blyth                127.0.0.1          
    wiki        Thunderbird                     ThunderbirdScreenSnapz001.png   26807       1197510085                                             blyth                127.0.0.1          
    wiki        XsltMacroTest                   info.xml                        285         1205135048    for Xslt testing                         blyth                127.0.0.1          
    wiki        XsltMacroTest                   format.xsl                      497         1205135068    for Xslt testing                         blyth                127.0.0.1         


"""
import logging, sys, re, os, collections, datetime
from env.doc.tabrst import Table
log = logging.getLogger(__name__)


ftime_ = lambda _:datetime.datetime.fromtimestamp(_).strftime('%Y-%m-%dT%H:%M:%S' )


class TracAttachment(object):
    """
    http://g4pb.local/tracs/workflow/ticket/16

    Are missing attachments message/link
    """
    def __init__(self, db):
        self.recs = db("select * from attachment")
        self.setPage()

    def setPage(self, typ=None, id_=None):
        self.typ = typ
        self.id_ = id_

    def get_partial_sql(self, typ_=None, id_=None):
        where = " and ".join(filter(None,[
                 "1",
                 "type = '%(typ)s' " % locals() if typ is not None else None,
                 "id = '%(id_)s' " % locals() if id_ is not None else None,
                ]))

        sql = "select * from attachment where %(where)s " % locals()
        return sql 

    def _get_tab(self):
        tab = Table(hdr=True)
        cols = ["doclnk", "size", "ftime", "description" ]
        tab.append(cols)
        for rec in self.recs:
            if self.typ is not None and rec["type"] != self.typ: continue 
            if self.id_ is not None and str(rec["id"]) != str(self.id_): continue 
            docref = "%(type)s/%(id)s/%(filename)s" % rec 
            doclnk = ":tracdocs:`%s`" % docref
            rec["ftime"] = ftime_(rec["time"])
            tab.append([doclnk] + [unicode(rec[k]) for k in cols[1:]])
        pass
        return tab
    tab = property(_get_tab)


    def __repr__(self):
        return "<TracAttachment attachments %d> " % ( len(self.recs) )

    def __unicode__(self):
        return unicode(self.tab)

    def __str__(self):
        return unicode(self).encode('utf-8')
 
 


if __name__ == '__main__':

    level = 'INFO'
    logging.basicConfig(level=getattr(logging, level)) 
    from env.sqlite.db import DB
    from env.web.cnf import Cnf
 
    cnf = Cnf.read("workflow_trac2sphinx", "~/.workflow.cnf") 
    db = DB(cnf["tracdb"], asdict=True)
    ta = TracAttachment(db)
    print ta

    ta.setPage(typ="wiki", id_="Trac2LatexTesting" )
    print ta
 
    ta.setPage(typ="ticket" )
    print ta
 


