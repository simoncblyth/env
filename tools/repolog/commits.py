#!/usr/bin/env python
"""
Prepare useful reports from the :file:`~/.env/svnlog.db` created by `svnlog-collect` that 
gathers all my SVN commit messages from all repositories over the past year.

Usage::

     commits.py 2013:11   # report all commits within the specified week in all repositories
     commits.py           # report all collected commits      

.. warning:: Handle revisions as opaque strings rather than integers, for future git/fossil/etc.. compatibility


Debugging::

    simon:~ blyth$ svnlog-db
    -- Loading resources from /Users/blyth/.sqliterc

    SQLite version 3.7.14.1 2012-10-04 19:37:12
    Enter ".help" for instructions
    Enter SQL statements terminated with a ";"
    sqlite> .tables
    commits  repos  
    sqlite> 
    sqlite> select date, strftime('%Y:%W', datetime(date,"unixepoch","localtime")) as week, rid, rev, msg  from commits order by date desc limit 3 ;
    date        week        rid         rev         msg                                                        
    ----------  ----------  ----------  ----------  -----------------------------------------------------------
    1367208613  2013:17     1           3695        update svnlog for more flexible functional asdict approach 
    1367207549  2013:17     1           3694        improve presentation and querying of the all-repo weekly re
    1367201583  2013:17     4           929         arc lighttpd webacces config and curl access               
    sqlite> 

"""
import os, logging
log = logging.getLogger(__name__)
from env.db.simtab import Table


time_ = lambda field, tfmt, label:"""
      strftime('%(tfmt)s',datetime(%(field)s,'unixepoch','localtime')) as %(label)s """ % locals()
 
# kludge to pluck the 2nd as label by using "))as integer" rather than spaced ")) as integer" 
weekday_ = lambda field, wfmt, label:"""
      case cast (strftime('%(wfmt)s',datetime(%(field)s,'unixepoch','localtime'))as integer)     
           when 0 then 'Sun'
           when 1 then 'Mon'
           when 2 then 'Tue'
           when 3 then 'Wed'
           when 4 then 'Thu'
           when 5 then 'Fri'
           else 'Sat' 
       end as %(label)s"""  % locals()


class QWeekly(object):
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


    def commitly_sql_(self):
        table = "commits"
        tfmt = "%Y/%m/%d %H:%M:%S"
        cols = ["rid as rid", "rev as rev", "date as date", "rid||':'||rev as ridrev", "msg as msg","details as details",time_("date",tfmt,"time"), weekday_("date","%w","weekday")] 
        labels = map(lambda _:_.split(" as ")[1].rstrip(), cols)  # caution keep the spacing around the " as "
        cols = ",".join(cols)
        sql = "select %(cols)s from %(table)s ;" % locals()
        return sql, labels

    def __init__(self, db, opts):
        self.db = db 
        self.opts = opts
        csql, clabels = self.commitly_sql_()
        #log.info(csql)
        cdict = {}
        for d in db.commits.iterdict(csql,",".join(clabels)):
            ridrev = "%s:%s"%(d['rid'],d['rev'])
            cdict[ridrev] = d
        self.cdict = cdict
        # keyed by the ridrev ie "1:500" repo 1 revision 500 

    def __call__(self, args, anno={}):
        log.debug("args %s " % args )
        sql, labels = self.weekly_sql_(self.db.repos)
        for dweek in self.db.commits.listdict( sql , labels=",".join(labels)):
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
                    ridrevs = dweek['revs'].split(",")
                    ridrevs = sorted(ridrevs,key=lambda _:self.cdict[_]['date'])
                    weekdays = map( lambda _:self.cdict[_]['weekday'] , ridrevs )
                    for day in 'Sun Mon Tue Wed Thu Fri Sat'.split():
                        print 
                        for ridrev in filter(lambda _:self.cdict[_]['weekday'] == day,ridrevs):
                            drev = self.cdict[ridrev]
                            assert drev['ridrev'] == ridrev, ridrev
                            print self.opts.commitfmt % drev
                            if self.opts.verbose: 
                                print  
                                print drev['details']
                pass     
            pass    
        return self


class Commits(object):
    """
    """
    def __init__(self, dbpath):
        self.repos = self.repos_(dbpath)
        log.debug("from %s find repos %s " % (dbpath, self.repos) )  
        self.commits = Table(dbpath, "commits") 
        self.lastrev = self.lastrev_()
        pass 

    def __repr__(self):
        return "%s %s" % ( self.__class__.__name__ , repr(self.lastrev)) 

    def repos_(self, dbpath):
        """
        :param path: to svnlog sqlite3 DB
        :return: dict like  `{1: 'env', 2: 'heprez', 3: 'dybsvn'}`
        """
        r = Table(dbpath, "repos" )
        repos = {}
        for url,id in r.asdict(lambda d:d["url"],lambda d:d["id"]).items():
            name = os.path.basename(url)
            repos[id] = name
        return repos

    def lastrev_(self):    
        """
        Query for last revisions from each repo. 
        NB resisted temptation to assume integer revision codes as that is not future safe.
        """
        last = {}
        for id,name in self.repos.items():
            sql  = "select rev from commits where rid=%(id)s order by date desc limit 1 " % locals()
            last[name] = self.commits.listdict(sql,labels="rev" )[0]['rev']
        return last

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
    op.add_option("-c", "--commitfmt", default="%(ridrev)-8s    %(time)s [%(weekday)s]     %(msg)s " )

    opts, args = op.parse_args()
    assert not(opts.verbose and opts.terse), "those options are incompatible"
    level = getattr( logging, opts.loglevel.upper() )
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)-8s %(message)s")
    return opts, args


def main():
    opts, args = parse_args(__doc__)
     
    anno = Annotate(opts.annopath)
    db = Commits(opts.dbpath)  
    print db
    q = QWeekly(db, opts)(args, anno=anno)


if __name__ == '__main__':
    main()



