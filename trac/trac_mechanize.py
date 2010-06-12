"""

     Form based Authorized Access to Trac Timeline RSS feed 
     with configurable cache_hours for the feed
     
        from env.trac.trac_mechanize import Timeline
        t = Timeline("dybsvn").init()
        print t.rss
        print len(t.rss.entries)
        print t.rss.entries[0]

"""
import os
import stat
from datetime import datetime

from mechanize import Browser
from ConfigParser import ConfigParser 
import feedparser
import cPickle as pickle


class Timeline(dict):
    def __init__(self, instance ):
        cfp = ConfigParser()
        cfp.read( os.path.expanduser("~/.mechanize/trac.ini") )
        self.update(cfp.items(instance))
        self.cachepath = os.path.expanduser("~/.mechanize/%s.pc" % instance )         

    def init(self):
        self.rss = None
        if os.path.exists(self.cachepath):
            st = os.stat(self.cachepath)
            age = datetime.now() - datetime.fromtimestamp( st[stat.ST_CTIME] )  
            if float(age.seconds) / 60 / 60 < self.get('cache_hours',3): 
                self.rss = pickle.load( file(self.cachepath,"r") ) 
        if not(self.rss):
            self.rss = self.get_timeline_rss()
            pickle.dump( self.rss , file(self.cachepath,"w") )
        return self

    def get_timeline_rss(self):
        self._login()
        return self._get_timeline_rss()

    def _login(self): 
        br = Browser()
        br.open(self["url"])
        br.follow_link(text="Login")
        br.select_form(nr=1)
        br['user'] = self["user"]  
        br['password'] = self["pass"]
        print br.form
        r = br.submit()
        print r 
        self.br = br

    def _get_timeline_rss(self):
        br = self.br
        br.follow_link(text="Timeline")
        print br.title()
        br.select_form(nr=1)
        br["daysback"] = "10"
        for c in [c for c in br.form.controls if c.type=="checkbox"]:c.items[0].selected = c.name in ['build','changeset']
        print br.form
        r = br.submit()
        print r 
        br.follow_link(text="RSS Feed")
        rss = feedparser.parse( br.response() )
        return rss

    def dump_rss(self):
        rss = self.rss
        print "entries : %s " % len(rss.entries)
        for e in rss.entries:
            print e.link, e.updated, e.date_parsed

if __name__=='__main__':
    import sys
    instance = len(sys.argv)>1 and sys.argv[1] or "env"
    tt = Timeline(instance)
    tt.dump_rss()

