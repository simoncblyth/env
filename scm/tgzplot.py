#!/usr/bin/env python
from __future__ import with_statement
import os, logging
log = logging.getLogger(__name__)
from datetime import datetime

from env.plot.highstock import HSOptions


class TGZPlot(object):
    def __init__(self, tgz ):
	self.tgz = tgz    

    def hsopts(self, node, select=[] ):
        """
	:param select: list of strings requires to be included in the names 
        """
        now = datetime.now().strftime("%c")
	hso = HSOptions()
        hso['series'] = []

        for _ in self.tgz.items(node):
            name, dir = _
            if len(select) == 0 or name in select: 
                data = self.tgz.data( node, _ )
	        hss = dict(name=name, data=data, tooltip=dict(valueDecimals=2)) 
	        hso['series'].append( hss )
		log.info("append series %s %s of length %s  " % (name,dir, len(data)) )     
            else:		
		log.info("skipping %s %s " % (name,dir) )     

	pfx = self.tgz.pfx(node)     # do here as will already be cached following data access
	hso['renderTo'] = "container_%s" % node
        hso['title'] = "%s %s %s" % ( node, pfx, now )
        return hso


    def jsondump(self, path, node=None, select=[]):
	"""
        :param path: in which to dump the json series 
	"""
        log.info("write json to %s " % path ) 
	hso = self.hsopts(node, select)
        with open(path,"w") as fp:
            fp.write(repr(hso))



if __name__ == '__main__':
    pass
    from env.scm.tgz import TGZ
    tgz = TGZ()
    plt = TGZPlot(tgz)
    hso = plt.hsopts('Z9:229')
    print hso 

