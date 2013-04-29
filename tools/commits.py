#!/usr/bin/env python
"""
Prepare useful reports from the :file:`~/.env/svnlog.db` created by `svnlog-collect` that 
gathers all my SVN commit messages from all repositories over the past year.

Usage::

     commits.py 2013:11   # report all commits within the specified week in all repositories
     commits.py           # report all collected commits      

.. warning:: Handle revisions as opaque strings rather than integers, for future git/fossil/etc.. compatibility

"""
import os, logging
log = logging.getLogger(__name__)
from env.db.simtab import Table

class QWeekly(object):

    def repos_(self, path):
        """
        :param path: to svnlog sqlite3 DB
        :return: dict like  `{1: 'env', 2: 'heprez', 3: 'dybsvn'}`
        """
        r = Table(path, "repos" )
        repos = {}
        for url,id in r.asdict(lambda d:d["url"],lambda d:d["id"]).items():
            repos[id] = os.path.basename(url)
        return repos

    def weekly_cols_(self, repos):
        """
        :param repos:
        :return: query column strings 

        Example of columns::

                 strftime('%Y:%W', datetime(date,'unixepoch','localtime') ) as week, 
                 group_concat(rid||":"||rev) as revs from commits 
                 
        """
        dt_ = lambda field,label:"datetime(%(field)s,'unixepoch','localtime') as %(label)s " % locals()
        ccw_ = lambda (condition,label):"count(case when %(condition)s then 1 end) as N%(label)s " % locals()
        week_ = lambda wfmt,field,label:'strftime(\'%(wfmt)s\', datetime(%(field)s,"unixepoch","localtime") ) as %(label)s ' % locals()
        gcon_ = lambda col1,col2:"group_concat(%(col1)s||':'||%(col2)s) as revs " % locals()
        count_ = lambda field,label:"count(%(field)s) as %(label)s" % locals()
        pass
        cols = [ week_("%Y:%W","date","week"), count_("*","N") ]   #
        cols += map(ccw_,  [("rid=%s" % k,v) for k,v in repos.items()]) 
        cols += [ gcon_("rid","rev") ]
        return cols

    def weekly_sql_(self, repos):
        """
        :param repos:
        :return: sql for weekly query, labels

        Query result dicts provide ridrev for each week and repo totals::

            {'week': '2012:17', 'revs': '1:3443', 'Nenv': 1, 'N': 1}
            {'week': '2012:19', 'revs': '1:3459,1:3458,1:3457,1:3456,1:3455,1:3454,1:3453,1:3452,1:3451,1:3450,1:3449,1:3448,1:3447,1:3446,1:3445,1:3444', 'Nenv': 16, 'N': 16}
            {'week': '2012:20', 'revs': '1:3465,1:3464,1:3463,1:3462,1:3461,1:3460', 'Nenv': 6, 'N': 6}
            {'week': '2012:21', 'revs': '1:3468,1:3467,1:3466', 'Nenv': 3, 'N': 3}

        """  
        table = "commits"
        cols = self.weekly_cols_(repos)
        labels = map(lambda _:_.split(" as ")[1].rstrip(), cols)  # caution keep the spacing around the " as "
        cols = ",".join(cols)
        sql = "select %(cols)s from %(table)s group by week order by week ;" % locals()
        return sql, labels 

    def __init__(self, opts):
        self.opts = opts
        self.repos = self.repos_(opts.dbpath)
        log.debug("from %s find repos %s " % (opts.dbpath, self.repos) )  
        commits = Table(opts.dbpath, "commits" )
        self.cdict = commits.asdict(lambda d:"%s:%s"%(d['rid'],d['rev']), lambda d:d ) # keyed by the ridrev ie "1:500" repo 1 revision 500 
        self.commits = commits

    def __call__(self, args, anno={}):
        log.debug("args %s " % args )
        sql, labels = self.weekly_sql_(self.repos)
        for dweek in self.commits.listdict( sql , labels=",".join(labels)):
            week = dweek['week']
            note = anno.get(week,{}).get('note',"")
            dweek.update(note=note)
            select = len(args) == 0 or week in args
            if select:
                if self.opts.terse:
                    print self.opts.titlefmt % dweek
                else:
                    print 
                    print self.opts.titlefmt % dweek
                    print 
                    for ridrev in dweek['revs'].split(","):
                        print "%-8s %s " % ( ridrev, self.cdict[ridrev]['msg'] )
                        if self.opts.verbose: 
                            print  
                            print self.cdict[ridrev]['details']
                pass     
            pass    
        return self



class Annotate(dict):
    def __init__(self, cnfpath ):
        """
        Read annotation config file into this dict

        :param cnfpath: config file path
        """
        from ConfigParser import ConfigParser
        cpr = ConfigParser()
        cpr.read(os.path.expanduser(cnfpath))
        for sect in cpr.sections():
            self[sect] = dict(cpr.items(sect))
         

def parse_args(doc):
    """
    Return config dict and commandline arguments 

    :param doc:
    :return: cnf, args  
    """
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-l", "--loglevel",   default="INFO", help="logging level : INFO, WARN, DEBUG ... Default %default"  )
    op.add_option("-d", "--dbpath",   default="~/.env/svnlog.db", help="Path to multi-repo collection of SVN commit messages. Default %default"  )
    op.add_option("-a", "--annopath",   default="~/.env/annoweek.cnf", help="Path to weekly annotation config file. Default %default"  )
    op.add_option("-v", "--verbose",  action="store_true", help="Details of the commits as well as messages and totals. Default %default"  )
    op.add_option("-t", "--terse",  action="store_true", help="Summary and totals only. Default %default"  )
    op.add_option("-f", "--titlefmt", default="**%(week)s**   e%(Nenv)-2s h%(Nheprez)-2s d%(Ndybsvn)-2s w%(Nworkflow)-2s     %(note)s   " )

    opts, args = op.parse_args()
    assert not(opts.verbose and opts.terse), "those options are incompatible"
    level = getattr( logging, opts.loglevel.upper() )
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)-8s %(message)s")
    return opts, args


def main():
    opts, args = parse_args(__doc__)
    anno = Annotate(opts.annopath)
    q = QWeekly(opts)(args, anno=anno)


if __name__ == '__main__':
    main()



