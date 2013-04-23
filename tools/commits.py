#!/usr/bin/env python
"""
Prepare useful reports from the :file:`~/.env/svnlog.db` created by `svnlog-collect` that 
gathers all my SVN commit messages from all repositories over the past year.

.. warning:: Handle revisions as opaque strings rather than integers, for future git/fossil/etc.. compatibility

select strftime('%Y/%W', datetime(date,'unixepoch','localtime') ) as W, count(*) as N, count(case when rid=1 then 1 end) as N1,count(case when rid=2 then 1 end) as N2, count(case when rid=3 then 1 end) as N3 from commits group by W order by W  ;

"""
import os
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

    def cols_(self, repos):
        """
        select strftime('%Y:%W', datetime(date,'unixepoch','localtime') ) as week, group_concat(rid||":"||rev) as revs from commits group by week order by week ; 
        """
        dt_ = lambda field,label:"datetime(%(field)s,'unixepoch','localtime') as %(label)s " % locals()
        ccw_ = lambda (condition,label):"count(case when %(condition)s then 1 end) as N%(label)s " % locals()
        week_ = lambda wfmt,field,label:'strftime(\'%(wfmt)s\', datetime(%(field)s,"unixepoch","localtime") ) as %(label)s ' % locals()
        gcon_ = lambda col1,col2:"group_concat(%(col1)s||':'||%(col2)s) as revs " % locals()
        count_ = lambda field,label:"count(%(field)s) as %(label)s" % locals()
        cols = [ week_("%Y:%W","date","week"), count_("*","N") ]   #
        cols += map(ccw_,  [("rid=%s" % k,v) for k,v in repos.items()]) 
        cols += [ gcon_("rid","rev") ]
        return cols

    def sql_(self, repos):
        table = "commits"
        cols = self.cols_(repos)
        labels = map(lambda _:_.split(" as ")[1].rstrip(), cols)  # caution keep the spacing around the " as "
        cols = ",".join(cols)
        sql = "select %(cols)s from %(table)s group by week order by week ;" % locals()
        return sql, labels 

    def __init__(self, path):
        repos = self.repos_(path)
        sql, labels = self.sql_(repos)
        self.repos = repos
        commits = Table(path, "commits" )
        commd = commits.asdict(lambda d:"%s:%s"%(d['rid'],d['rev']), lambda d:d['msg'] ) 
        print sql 
        for week in commits.listdict( sql , labels=",".join(labels)):
            print 
            print week
            for ridrev in week['revs'].split(","):
                print ridrev, commd[ridrev]
    

if __name__ == '__main__':
    q = QWeekly("~/.env/svnlog.db")




