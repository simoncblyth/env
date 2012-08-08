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
    maxage = 1   # maximum allowable tarball age in days, typically 1 day  
    msday = 24*60*60*1000 
    smrycol = ('name','ltime','lsize','ldays','ldate')
    statcol = ("node","nok","nwarn","nalarm","status")

    def __init__(self, hub=None):
        self.hub = hub
        self.nodes = []
        self.smry = {}
        self.stat = {}

    def _summary(self, tgz, node ):
        """
        Queries sqlite db to collect the last tgz status for all
        items for the node

        :param tgz:
	:param node:
	:return: list of dict summarizing tarballs for each item 
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
        """
        :param node:
        :param smry:
        :return: status dict for the node  

        Out of range values are annotated via promotion of 
        simple values (numbers and strings) into dicts.
        This is inplace editing the retained smry dict 
        """
	nok = 0
	nwarn = 0
	nalarm = 0

        for d in smry:
	    if d['ldays'] > self.maxage:		
		 nalarm += 1
                 d['ldays'] = dict(v=d['ldays'], msg="overage", st="alarm")
                 d['name']  = dict(v=d['name'], msg="overage", st="alarm")
            else:
		 nok += 1   

	st = "ok"
        if nwarn > 0:
	    st = "warn"
	if nalarm > 0:
	    st = "alarm"

        stat = {}
        stat['node'] = dict(v=node,st=st)
        stat['nok'] = nok
        stat['nwarn'] = nwarn
        stat['nalarm'] = nalarm
        stat['status'] = st
        return stat

    def collect_summary(self, tgz, node ):
        """
        :param tgz:
        :param node: 
        """
	smry = self._summary(tgz, node)
        stat = self._status(node, smry )
        pass
        self.nodes.append(node)
        self.smry[node] = smry 
        self.stat[node] = stat 

    def _conclusion(self):
        """
        """ 
        sts = []
        for stat in self.status():
            if stat['status'] == "ok":
                pass
            else:
                sts.append(stat['status'])
        return ":".join(sts) if len(sts) > 0 else "ok" 
    conclusion = property(_conclusion)

    def status(self):
        return map(lambda _:self.stat[_], self.nodes)

    def __str__(self):
        return "\n".join( map(pformat, self.smry.items() + self.stat.items()))



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)	

    from env.scm.tgz import TGZ

    tgz = TGZ()
    stat = TGZStat(hub="C2")
    stat.collect_summary( tgz, "C" )
    print stat
    print "conclusion:", stat.conclusion

    

