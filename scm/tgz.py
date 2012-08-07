#!/usr/bin/env python
"""

"""
import os, logging
log = logging.getLogger(__name__)
from pprint import pformat
from datetime import datetime
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
	"""
	:param node: tag of remote node
	:return: list of distinct directories for the remote node
	"""
        dirs = map(lambda _:_[0], self.tab("select distinct(dir) from %s where node='%s' " % (self.tn, node)))
        return dirs

    def items(self, node ):
	"""
	:param node: tag of remote node
	:return: list of 2-tuples with (short name with common prefix removed,absolute dir)

	For example::

		(u'repos/data', u'/volume1/var/scm/backup/g4pb/repos/data')
		(u'tracs/data', u'/volume1/var/scm/backup/g4pb/tracs/data')

	"""
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

    def data(self, node, item , xsql="" ):
        """
	:param node: 
	:param item: specifier tuple as returned by items
	:param xsql: extra SQL to tack on to the query, eg "desc limit 1 " to get the last value
	:return: list of 2 element lists with the time series of tarball sizes in date order
        """
        name, dir = item
        data = []
        sql = "select strftime('%s',date)*1000, size from %s where dir='%s' and node='%s' order by date %s " % ( "%s", self.tn, dir, node, xsql )
        for d in self.tab(sql):
	    l = list(d)	
	    if l[1]<11.:l[1] = l[1]*10.    # arbitaryish scaling for visibiity		  
            data.append(l)
        return data

    def dump(self, node):
	"""    
	:param node: 

	debug dumping 
        """
        for _ in tgz.items(node):
  	    name, dir = _
	    print name
	    data = self.data(node, _)
	    print data
        pass


if __name__ == '__main__':


    logging.basicConfig(level=logging.INFO)	
    tgz = TGZ()

    node = 'Z9:229'

    dirs = tgz.dirs(node)
    print "\n".join(dirs)
    for _ in tgz.items(node):
        print _


