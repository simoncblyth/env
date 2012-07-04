#!/usr/bin/env python
"""
Frontend for running Fabric monitor

"""
import os, logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from fabric.api import env
from env.tools.libfab import fabloop, rrun
from env.scm.tgz import TGZ
from env.scm.tgzplot import TGZPlot


def monitor(node):
    """
    Workaround for fabric strictures by writing to DB separately for each remote and 
    benefiting from the uniqing (re-runability) built into the SQL. 

    TODO:

    #. handle failed connection presentationally [current time and a -1 ?]
    #. configure this hub dependant things : specifier g4pb/cms02 + selection  ?

    """
    tn    = "tgzs"
    srvnode = "g4pb"     
    select = ["%s/%s" % ( type, proj) for type in ("tracs","repos","svn") for proj in ("env","heprez","dybaux","dybsvn","tracdev","aberdeen","workflow")]

    dbp   = os.path.expandvars("$LOCAL_BASE/env/scm/scm_backup_monitor.db")   
    jsonp = os.path.expandvars("$APACHE_HTDOCS/data/scm_backup_monitor_%(node)s.json" % locals() ) 
    assert os.path.exists(os.path.dirname(jsonp)), jsonp

    tgz = TGZ(dbp, tn)
    cmd = tgz.cmd % locals()
    ret = rrun(cmd)
    if ret:
        tgz.parse(ret.split("\r\n"), node )  

    plt = TGZPlot(tgz)
    plt.jsondump(jsonp, node=env.host_string, select=select)

    pass
    log.info("to check:  echo .dump %s | sqlite3 %s  " % (tn,dbp) )
    log.info("to check: cat %s | python -m simplejson.tool " % jsonp )


def main():
    import sys	
    if len(sys.argv) > 1:
        khosts = sys.argv[1]
    else:
	khosts = os.environ['NODE_TAG']    
    fabloop( khosts, monitor )


if __name__ == '__main__':
    main()	

