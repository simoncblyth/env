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
from env.doc.tabledoc import TabularData


class TGZSmry(object):

    def __init__(self, stat ):
        self.stat = stat

    def _annotate(self, node, stat ):
	"""
	:param node:

	Inplace annotates the collected summary dict based on allowable limits 
	and appends to the collective per-node status list 

	Seems that docutils is converting backticks into ordinary quotes 
	"""
        f = {}
        f['ok']    = lambda _:r":ok:`%s`" % _
        f['alarm'] = lambda _:r":alarm:`%s`" % _
        f['warn']  = lambda _:r":warn:`%s`" % _

    def summary_table(self, node ):
        smry = self.stat.smry[node]
	self.annotate_smry( node, smry )
        tsmry = TabularData(smry)
        rst = tsmry.as_rst(cols=self.smrycol)
	rst = rst.replace("'","`")	
	return rst

    def status_table(self):
        tstat = TabularData(self.stat.status)
        rst = tstat.as_rst(cols=self.stat.statcol)
	rst = rst.replace("'","`")	
	return rst


    def node_summary(self, node, table=""):
        tmpl = r"""
%(node)s
~~~~~~~~~

%(table)s

* `scm_backup_monitor_%(node)s.json </data/scm_backup_monitor_%(node)s.json>`_

.. stockchart:: /data/scm_backup_monitor_%(node)s.json container_%(node)s

        """
        return tmpl % locals()


    def hub_summary(self, tgz, nodes):
        head = r"""

.. include:: /sphinxext/roles.txt

NTU (hub C2)
-------------
  
""" 
        body = ""
        for node in nodes:
            table = self.summary_table(tgz, node)
            body += self.node_summary( node, table=table )

        stat = self.status_table()
	return head + stat + "\n" + body


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)	

    from env.scm.tgz import TGZ
    smy = TGZSmry()
    tgz = TGZ()
    sta = TGZStat()

    node = 'Z9:229'
    print smy.hub_summary( tgz, [node] )


 
