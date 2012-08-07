#!/usr/bin/env python
"""
"""
import os, logging
log = logging.getLogger(__name__)
from pprint import pformat
from datetime import datetime

class TGZStat(object):
    """
    """
    maxage = 25   # maximum allowable tarball age in days, typically 1 day  
    msday = 24*60*60*1000 
    smrycol = ('name','ltime','lsize','ldays','ldate')
    statcol = ("node","nok","nwarn","nalarm","status")

    def __init__(self):
        self.smry = {}
        self.stat = {}
	self.status = []

    def _summary(self, tgz, node ):
        """
        Queries sqlite db to collect the last tgz status for all
        items for the node

        :param tgz:
	:param node:
	:return: list of dict summarizing tarballs from remote backup node
	"""
        now = int(datetime.now().strftime("%s"))*1000
        smry = []
        for _ in tgz.items(node):
	    name, dir = _
	    last = tgz.data(node, _, "desc limit 1")
	    assert len(last) == 1, len(last)
            ltime,lsize = last[0]
	    ldays = float(now - ltime)/float(self.msday)
	    ldate = datetime.utcfromtimestamp(ltime/1000)
	    smry.append(dict(zip(self.smrycol,[name,ltime,lsize,ldays,ldate])))
        pass
        return smry

    def _status(self, node, smry ):

	nok = 0
	nwarn = 0
	nalarm = 0

        for d in smry:
	    if d['ldays'] > self.maxage:		
		 nalarm += 1
            else:
		 nok += 1   

	st = "ok"
        if nwarn > 0:
	    st = "warn"
	if nalarm > 0:
	    st = "alarm"

        stat = {}
        stat['node'] = node
        stat['nok'] = nok
        stat['nwarn'] = nwarn
        stat['nalarm'] = nalarm
        stat['status'] = st
        return stat

    def add_summary(self, tgz, node ):
        """
        Using convention that methods beginning with a single underscore set no state
        """
	smry = self._summary(tgz, node)
        stat = self._status(node, smry )
        pass
        self.smry[node] = smry 
        self.stat[node] = stat 


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)	

    from env.scm.tgz import TGZ

    stat = TGZStat()
    tgz = TGZ()

    node = "C"
    stat.add_summary( tgz, node )
    print pformat(stat.smry[node]) 
    print pformat(stat.stat[node]) 


