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
from env.doc.tabledoc import AnnotatedTabularData


class HubRst(list):
    tmpl = r"""

.. include:: /sphinxext/roles.txt

%(hub)s hub : %(conclusion)s
-----------------------------------
 
.. contents:: :local:

 
""" 
    def __init__(self, **kwa ):
        self.kwa = kwa
        list.__init__(self)

    def __str__(self):
	return "\n".join( [self.tmpl % self.kwa] + map(str, self) )


class NodeRst(dict):
    tmpl = r"""
%(node)s : %(status)s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

%(table)s

* `scm_backup_monitor_%(node)s.json </data/scm_backup_monitor_%(node)s.json>`_

.. stockchart:: /data/scm_backup_monitor_%(node)s.json container_%(node)s

    """
    def __str__(self):return self.tmpl % self






class TGZSmry(object):
    """
    Presentation of TGZStat 
    """ 
    def __init__(self, stat ):
        """
        :param stat: TGZStat instance
        """
        self.stat = stat

    def node_table(self, node ):
        smry = self.stat.smry[node]
        tsmry = AnnotatedTabularData(smry)
        return tsmry.as_rst(cols=self.stat.smrycol,annonly=True)

    def status_table(self):
        status = self.stat.status()
        tstat = AnnotatedTabularData(status)
        return tstat.as_rst(cols=self.stat.statcol,annonly=False)

    def hub_summary(self):
        hr = HubRst(hub=self.stat.hub, conclusion=self.stat.conclusion)
        stat_table = self.status_table()
        hr.append( stat_table )
        for node in self.stat.nodes:
            node_table = self.node_table(node)
            nr = NodeRst(node=node, table=node_table, status=self.stat.stat[node]['status'] )
            hr.append(nr) 
        pass
        return hr     



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)	

    from env.scm.tgz import TGZ
    from env.scm.tgzstat import TGZStat

    tgz = TGZ()

    stat = TGZStat(hub="C2")
    stat.collect_summary( tgz, "C" )
    #print stat

    smry = TGZSmry(stat)
    print smry.hub_summary()


 
