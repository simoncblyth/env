#!/usr/bin/env python
"""
Collects authors within a set of Trac+SVN repos into sqlite3 DB 

Usage
~~~~~~

::

   ./svnauthors.py --help
   ./svnauthors.py read      # from configured urls
   ./svnauthors.py ls        # list entries, reading from DB using python 
   ./svnauthors.py dbg       # list entries, reading from DB using sqlite3 at command line level
   ./svnauthors.py git       # list entries reading from DB in git authorlist format
   ./svnauthors.py git > ~/svnauthors.txt

Config
~~~~~~~~
::

    [svnauthors]

    url = http://dayabay.phys.ntu.edu.tw/tracs/%s/report/11?format=csv
    repos = env heprez tracdev aberdeen
    dbpath = ~/.env/svnauthors.db
    table = authors

Debug
~~~~~

::

   echo "select * from authors ;" | sqlite3 ~/.env/svnauthors.db 

Issues 
~~~~~~~

#. lacks authenticated access, so the report has to be publically accessible

"""
import os, logging
from pprint import pformat
from datetime import datetime
log = logging.getLogger(__name__)
from ConfigParser import ConfigParser

from env.db.simtab import Table
import csv
import urllib2


def read_csv(url):
    """
    :param: url returning csv formatted table
    :return: list of dicts 

       {'Account': 'blyth', 'Name': 'simon blyth', 'Last_visit': '2013-04-01 06:57:52', 'Email': 'blyth@hep1.phys.ntu.edu.tw'}
    """ 
    req = urllib2.Request( url )
    opener = urllib2.build_opener()
    try:
        f = opener.open(req)
    except urllib2.HTTPError:
        log.warn("HTTPError opening %s " % url )
        f = None
    return csv.DictReader(f, delimiter=',') if f else []

class Cnf(dict):
    def __init__(self, sect, cnfpath="~/.env.cnf" ):
        cpr = ConfigParser()
        cpr.read(os.path.expanduser(cnfpath))
        self.update(cpr.items(sect)) 
        self['sect'] = sect
        self['sections'] = cpr.sections()

def parse_args(doc):
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-c", "--cnfpath",   default="~/.env.cnf", help="path to config file Default %default"  )
    op.add_option("-l", "--loglevel",   default="INFO", help="logging level : INFO, WARN, DEBUG ... Default %default"  )
    op.add_option("-s", "--sect",      default="svnauthors", help="section of config file... Default %default"  )
    opts, args = op.parse_args()
    loglevel = getattr( logging, opts.loglevel.upper() )
    logging.basicConfig(level=loglevel)
    cnf = Cnf(opts.sect, opts.cnfpath)
    log.debug("reading config from sect %s of %s :\n%s " % (opts.sect, opts.cnfpath, cnf))  
    return cnf, args

class GitAuthor(dict):
    __str__ = lambda _:"""%(Account)s = %(Name)s <%(Email)s>""" % _

class Authors(object):
    def __init__(self, cnf):
        tab = Table(cnf['dbpath'], cnf['table'], Email="text", Account="text", Name="text", Last_visit="date"  )
        self.tab = tab
        self.cnf = cnf
        pass  

    def read(self, repos=None):
        if not repos:
            repos=self.cnf['repos']
        for repo in repos.split():
            self.read_one(repo)

    def read_one(self, repo):
        tab = self.tab 
        url = self.cnf['url'] % repo
        for d in read_csv(url):
            if d.has_key('Email') and d.has_key('Account') and d.has_key('Name') and d.has_key('Last_visit'):
                log.info(d) 
                tab.append( d )  
            else:
                log.warn("skip incomplete dict %s " % d )
        tab.insert()
        tab[:] = []

    def ls(self):
        for d in self.tab("select * from %(table)s ;" % self.cnf):
            print d

    def git(self):
        for d in map(GitAuthor, self.tab.iterdict("select * from %(table)s order by Last_visit ;" % self.cnf)):
            if d['Email'] == 'None':continue
            print d

    def dbg(self):
        cmd = "echo \"select * from %(table)s ;\" | sqlite3 %(dbpath)s " % self.cnf
        for line in os.popen(cmd).readlines():
            print line,


    def __call__(self, args):
        for arg in args:
            if arg == 'read':
                self.read()
            elif arg == 'ls':
                self.ls()
            elif arg == 'git':
                self.git()
            elif arg == 'dbg':
                self.dbg()


if __name__ == '__main__':
    cnf, args = parse_args(__doc__)
    Authors(cnf)(args)





