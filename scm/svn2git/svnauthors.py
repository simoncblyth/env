#!/usr/bin/env python
"""
Collects authors within a set of Trac+SVN repos into sqlite3 DB 

Prepare authors file
----------------------

Git associates commits with email addresses rather than user named like SVN.
So need to prepare a mapping file

Using trac report 11 is csv format 

  * http://dayabay.phys.ntu.edu.tw/tracs/env/report/11?format=csv 
 
  * :env:`trunk/scm/svnauthors.py`

::

   ~/env/scm/svnauthors.py read                    # reads the trac report 11 and inserts into sqlite3 DB
   ~/env/scm/svnauthors.py git > ~/svnusers.txt    # reads from DB, dumping in git author format



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
#. re-reading causes duplication, need to establsh iddentity without primary keying email, which is too draconian


"""
import os, logging
log = logging.getLogger(__name__)

from pprint import pformat
from datetime import datetime
from config import parse_args

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
        return "\n".join([str(d) for d in self.tab("select * from %(table)s ;" % self.cnf)])

    def git(self):
        return "\n".join([str(d) for d in filter(lambda ga:ga['Email'] != 'None',map(GitAuthor, self.tab.iterdict("select * from %(table)s order by Last_visit ;" % self.cnf)))])

    def dbg(self):
        cmd = "echo \"select * from %(table)s ;\" | sqlite3 %(dbpath)s " % self.cnf
        return os.popen(cmd).read()

    def __call__(self, args):
        for arg in args:
            meth = getattr(self, arg, None) 
            if meth is None:
                log.warn("no method %s " % arg )
            else:
                log.info("call method %s " % arg )
                return meth() 

if __name__ == '__main__':
    cnf, args = parse_args(__doc__)
    print Authors(cnf)(args)





