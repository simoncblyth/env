#!/usr/bin/env python
"""
Provides the checker classes eg ``GZCheck`` used from the fabric fabfile to 
run remote commands and parse the responses and persist into sqlite DB 

#. keep fabric specifics in the fabfile 
#. minimize remote connections by

    #. doing a single remote find to get the paths and sizes   
    #. pulling datetime info encoded into path rather than querying remote file system.

Dependencies (not needed for 2.6+)

  #. ``pip install simplejson``


"""
from __future__ import with_statement
import os, re, logging
log = logging.getLogger(__name__)

from datetime import datetime
from simtab import Table
from jsonify import jsonify, jsdict



try:
    import json
except ImportError:
    import simplejson as json


class Path(str):
    """	
    paths of interest are assumed to contain a datetime encoding string such as 2012/02/21/093002
    NB pattern extent and strptime fmt extent must correspond precisely to allow correct parsing of
    string into datetime

    #. ``dir`` of the path is the portion before the encoded datetime

    """
    ptn = re.compile("\/\d{4}\/\d{2}\/\d{2}\/\d{6}\/")
    fmt = "/%Y/%m/%d/%H%M%S/"

    def __init__(self, path):
	str.__init__(self, path)    
	m = self.ptn.search(path)
	assert m, "failed to match path %s " % path 
	start, end = m.span()
        self.dir = path[:start]
	self.dat = path[start:end]
	self.aft = path[end:]
        dt  = datetime.strptime(self.dat,self.fmt) 
        self.dt = dt 
	self.date = dt.strftime("%Y-%m-%dT%H:%M:%S")   # SQLite can handle this 

    def __repr__(self):
	return "%s %s(%s)%s [%s]" % ( self.__class__.__name__, self.dir, self.dat, self.aft, self.date )     


class GZCheck(object):
    def __init__(self, dbpath, tn , srvnode ):
        """
        :param dbpath: path to SQLite db
        :param tn: name of DB table   

        """

        log.info("opening DB %s " % dbpath )
        self.cmd = "find $SCM_FOLD/backup/%s -name '*.gz' -exec du --block-size=1M {} \;" % srvnode
        self.tab = Table(dbpath, tn, nodepath="text primary key", node="text", dir="text", date="text", size="real" )
        self.tn = tn   

    def __call__(self, lines, node ):
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

    def check(self):
        self._check_current()

    def _check_current(self):
        """
        Checking the current entries, without using persisted prior entries

        NB despite the use of ``tab`` this is just the current in memory entries 

        #. group paths according to their folder, ie string before the date

        """
        dex = {}
        for d in self.tab:  
            dir = d['dir']
            if dir not in dex:
                dex[dir]=[]
            dex[dir].append(d)

        # dump paths within each folder ordered by date
        for k in dex.keys():
            print k
	    for p in sorted(dex[k],key=lambda _:_['date'] ):
                print repr(p)		

    def options(self, node, select ):
        """
	:param select: list of strings requires to be included in the names 

        """
        dirs = map(lambda _:_[0], self.tab("select distinct(dir) from %s where node='%s' " % (self.tn, node)))
        pfx = os.path.commonprefix(dirs)
        nams = map(lambda _:_[len(pfx):], dirs)
        log.info("commonprefix %s select %s " % (pfx, repr(select)) )


        now = datetime.now().strftime("%c")

	hso = HSOptions()
        hso['title'] = "%s %s %s" % ( node, pfx, now )
	hso['renderTo'] = "container_%s" % node
        hso['series'] = []

        for name,dir in zip(nams,dirs):
            if len(select) == 0 or name in select: 
                data = []
                sql = "select strftime('%s',date)*1000, size from %s where dir='%s' and node='%s' order by date" % ( "%s", self.tn, dir, node )
                for d in self.tab(sql):
		    l = list(d)	
	            if l[1]<11.:l[1] = l[1]*10.    # arbitaryish scaling for visibiity		  
                    data.append(l)

                #hss = HSSeries(name=name, data=data)
	        hss = dict(name=name, data=data, tooltip=dict(valueDecimals=2)) 
	        hso['series'].append( hss )
		log.info("append series %s %s of length %s  " % (name,dir, len(data)) )     
            else:		
		log.info("skipping %s %s " % (name,dir) )     

        return hso

    def jsondump(self, path, node=None, select=[]):
	"""
        :param path: in which to dump the json series 
	"""
        log.info("write json to %s " % path ) 
	hso = self.options(node, select)
        with open(path,"w") as fp:
            fp.write(repr(hso))



class HSSeries(dict):
    """
    Not working 
    """
    js = jsonify(r'''
            {
                  name : %(name)s,
		  data : %(data)s,
                tooltip : {
	                      valueDecimals : 2
			  },
	    }
              ''')
    def __repr__(self):
	return self.js % jsdict(self)    
 

class HSOptions(dict):
    """
    Dict containing options to be converted to JSON and passed
    to javascript function via jQuery ajax call::

	window.chart = new Highcharts.StockChart(options);

    """
    js = jsonify(r'''
             {
                        chart : {
                                renderTo : %(renderTo)s
                        },

                        rangeSelector : {
                                selected : 1
                        },

                        title : {
                                text : %(title)s
                        },
                        
                        series : %(series)s
             }
	     ''')
    def __repr__(self):
	return self.js % jsdict(self)    
    




if __name__ == '__main__':
    pass
    t = Table("scm_backup_check.db", "tgzs")
    print t.fields



