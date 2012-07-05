#!/usr/bin/env python
"""

"""
import os, logging
log = logging.getLogger(__name__)
from env.db.simtab import Table
from env.scm.datepath import Path

class TGZ(object):
    """
    Interface to backup tarball SQLite DB table, encompassing 

    #. remote command to run with Fabric to grab tarball sizes
    #. querying DB table to extract time series data (eg for plotting)

    """
    def __init__(self, dbpath=None, tn="tgzs" ):
        """
        :param dbpath: path to SQLite db
        :param tn: name of DB table   
        """
	if not dbpath:
            dbpath = os.path.expandvars("$LOCAL_BASE/env/scm/scm_backup_monitor.db")   
        if not os.path.exists(os.path.dirname(dbpath)):
            os.makedirs(os.path.dirname(dbpath))

        log.info("opening DB %s " % dbpath )
        self.cmd = "find $SCM_FOLD/backup/%(srvnode)s -name '*.gz' -exec du --block-size=1M {} \;"
        self.tab = Table(dbpath, tn, nodepath="text primary key", node="text", dir="text", date="text", size="real" )
        self.tn = tn   
	self._pfx = {}

    def parse(self, lines, node ):
        """
        Parse the response from the remote command and update local database,    
	new entries are added, changed entries (based on ``nodepath`` identity) 
	replace older ones.

        :param lines: list of strings response from the cmd
        :param node: ssh tag or alias of remote node on which the command was performed  
        :param roles: list of roles, using to correspond to hub node 

        """
        log.info("%s lines from node %s " % ( len(lines), node ) ) 
        for line in lines: 
            fields = line.split("\t")
            assert len(fields) == 2, "unexpected field count : %s " % repr(fields)
            size, path_ = fields
            path = Path(path_)
            nodepath= "%s:%s" % ( node, path_ )
            self.tab.add( node=node, nodepath=nodepath, size=size, dir=path.dir, date=path.date )
            pass 
        self.tab.insert()    

    def pfx(self, node):
	if not self._pfx.get(node, None):    
	    dirs = self.dirs(node)    
            pfx = os.path.commonprefix(dirs)
            self._pfx[node] = pfx
	return self._pfx[node]

    def dirs(self, node):
        dirs = map(lambda _:_[0], self.tab("select distinct(dir) from %s where node='%s' " % (self.tn, node)))
        return dirs

    def items(self, node ):
	dirs = self.dirs(node)    
        pfx = os.path.commonprefix(dirs)
	self._pfx[node] = pfx
	nams = map(lambda _:_[len(pfx):], dirs)
        log.info("commonprefix %s  " % (pfx) )
	return zip(nams,dirs)


    def okdata(self, node):
	"""    
        Tarballs in the past 10 days


        sqlite> select date(date),date from tgzs ;      date is not a good name for a field

        select * from tgzs where strftime('%s',date('now','-10 day')) < strftime('%s',date) ;

        select strftime('%s',date)*1000,count(*) as N,    from %s where node='%s' group by ( strftime('%s',date) - strftime('%s',date('now')) )/86400   ;

        """
        data = []
	sql = "select strftime('%s',date(date))*1000,count(*) from %s where node='%s' group by date(date) " % ( "%s", self.tn, node ) 
        for d in self.tab(sql):
	    l = list(d)	
            data.append(l)
        return data


    def data(self, node, item ):
        """
	:param node: 
	:param item: specifier tuple as returned by items
	:return: list of 2 element lists 
        """
        name, dir = item
        data = []
        sql = "select strftime('%s',date)*1000, size from %s where dir='%s' and node='%s' order by date" % ( "%s", self.tn, dir, node )
        for d in self.tab(sql):
	    l = list(d)	
	    if l[1]<11.:l[1] = l[1]*10.    # arbitaryish scaling for visibiity		  
            data.append(l)
        return data


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)	
    tgz = TGZ()
    node = 'Z9:229'
    for _ in tgz.items(node):
	print _[0]   
	data = tgz.data(node, _)
	print data

