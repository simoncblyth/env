#!/usr/bin/env python
"""
On G needs py26::

    python2.6 tgz.py


#. imposes limits and color the table red/green accordingly
#. integrate with sendmail notification + smry health cut to decide if notification needed 
#. include the summary table on the monitoring pages : maybe simply rst inclusion or a sphinxext to generate that on build ?
#. need node health and overall health determinations

ISSUES

#. suspect backtick escaping issue 


"""
import os, logging
log = logging.getLogger(__name__)
from datetime import datetime
from env.doc.tabledoc import TabularData

class TGZSmry(object):

    maxage = 25   # maximum allowable tarball age in days, typically 1 day  
    msday = 24*60*60*1000 
    smrycol = ('name','ltime','lsize','ldays','ldate')
    statcol = ("node","nok","nwarn","nalarm","status")

    def __init__(self, tgz ):
	self.tgz = tgz    
	self.status = []

    def summary(self, node):
        """
	:param node:
	:return: list of dict summarizing tarballs from remote backup node
	"""
        now = int(datetime.now().strftime("%s"))*1000
        smry = []
        for _ in self.tgz.items(node):
	    name, dir = _
	    last = self.tgz.data(node, _, "desc limit 1")
	    assert len(last) == 1, len(last)
            ltime,lsize = last[0]
	    ldays = float(now - ltime)/float(self.msday)
	    ldate = datetime.utcfromtimestamp(ltime/1000)
	    smry.append(dict(zip(self.smrycol,[name,ltime,lsize,ldays,ldate])))
        pass
        return smry

    def annotate_smry(self, node, smry):
	"""
	:param node:
	:param smry: summary dict 

	Inplace annotates the summary dict based on allowable limits 
	and appends to the collective per-node status list 

	Seems that docutils is converting backticks into ordinary quotes 
	"""
        ok    = lambda _:r":ok:`%s`" % _
        alarm = lambda _:r":alarm:`%s`" % _
        warn  = lambda _:r":warn:`%s`" % _

	nok = 0
	nwarn = 0
	nalarm = 0

        for d in smry:
	    if d['ldays'] > self.maxage:		
                 d['name'] = alarm(d['name'])
		 d['ldays'] = alarm(d['ldays'])
		 nalarm += 1
            else:
		 nok += 1   

        # decide on ok/warn/alarm status
        if nok > 0:
	    fn = ok
	    st = "ok"
        if nwarn > 0:
            fn = warn
	    st = "warn"
	if nalarm > 0:
	    fn = alarm
	    st = "alarm"

        stat = {}
        stat['node'] = fn(node)
        stat['nok'] = fn(nok)
        stat['nwarn'] = fn(nwarn)
        stat['nalarm'] = fn(nalarm)
        stat['status'] = fn(st)

        self.status.append(stat)


    def summary_table(self, node, backtick="'"):
	smry = self.summary(node)
	self.annotate_smry( node, smry )
        tsmry = TabularData(smry)
        rst = tsmry.as_rst(cols=self.smrycol)
	if backtick:
	   rst = rst.replace(backtick,"`")	
	return rst

    def status_table(self, backtick="'"):
        tstat = TabularData(self.status)
        rst = tstat.as_rst(cols=self.statcol)
	if backtick:
	   rst = rst.replace(backtick,"`")	
	return rst

    def rst_tables(self, nodes):
	rst = {}    
	rst['roles'] = ".. include:: /sphinxext/roles.txt\n\n" 
        for node in nodes:
            rst[node] = self.summary_table(node)
        rst['all'] = self.status_table()
	return rst['roles'] + rst['all'] + "\n" + "\n".join( map(lambda _:rst[_], nodes)) 

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)	

    from env.scm.tgz import TGZ
    tgz = TGZ()
    tgzsmry = TGZSmry(tgz)

    node = 'Z9:229'
    print tgzsmry.rst_tables( [node] )


 
