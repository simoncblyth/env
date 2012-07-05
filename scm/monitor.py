#!/usr/bin/env python
"""
Frontend for running Fabric monitoring
"""
import os, logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from fabric.api import env
from env.tools.libfab import fabloop, rrun
from env.tools.cnf import Cnf
from env.scm.tgz import TGZ
from env.scm.tgzplot import TGZPlot
from pprint import pformat
   

def writable(path):
    path = os.path.expandvars(path)
    if os.path.exists(path) and not os.access(path, os.W_OK):
        return False
    return True

def cnf_(hub, smc="~/.scm_monitor.cnf"):
    """	
    :param hub: tag of hub node, typically C2R, G or ZZ
    :return: dict config section for the hub

    Loads hub specific config parameters from ``~/.scm_monitor.cnf`` of format::

	[C2R]
	srvnode = cms02
	dbpath = $LOCAL_BASE/env/scm/scm_backup_monitor.db
	jspath = $APACHE_HTDOCS/data/scm_backup_monitor_%(node)s.json
	select = repos/env tracs/env repos/aberdeen tracs/aberdeen repos/tracdev tracs/tracdev repos/heprez tracs/heprez

    NB before the config dict arrives as parameter of monitor, which is invoked per remote node, additonal
    keys are added such as HOST

    Note the context keyholing around the fabloop in order to workaround the fabric features.
    """
    cnf = Cnf()
    r = cnf.read(smc)  
    if len(r) == 0:
        raise Exception("Config file does not exist at  %s " % smc )
    hubcnf = cnf.sectiondict(hub)

    path = hubcnf['dbpath']
    if not writable(path):
        raise Exception("cannot write to path %s " % path )

    hubcnf['HUB'] = hub
    return hubcnf


def monitor(cfg):
    """
    :param cfg: dict containing both hub and node config

    Task invoked by the fabloop for each remote node configured for the hub
    NB config is spread between:

    #. ~/.scm_monitor.cnf hub specific
    #. ~/.libfab.cnf node specific

    Workaround for fabric strictures by writing to DB separately for each remote and 
    benefiting from the uniqing (re-runability) built into the SQL. 

    TODO:

    #. handle failed connection presentationally [current time and a -1 ?]

    """
    print "monitor cfg: %s " % pformat(cfg) 
    node = cfg['HOST']
    assert node == env.host_string 
    tn    = "tgzs"
    srvnode = cfg['srvnode']   
    select = cfg['select'].split()
    dbp = os.path.expandvars(cfg['dbpath'])
    jsp = os.path.expandvars(cfg['jspath'] % dict(node=cfg['HOST']))
    assert os.path.exists(os.path.dirname(jsp)), jsp
    if not writable(jsp):
        raise Exception("jsp %s is not writable " % jsp)

    tgz = TGZ(dbp, tn)
   
    pull = True
    if pull:
        cmd = tgz.cmd % locals()
        ret = rrun(cmd)
        if ret:
            tgz.parse(ret.split("\r\n"), node )  

    plt = TGZPlot(tgz)
    plt.jsondump(jsp, node=env.host_string, select=select)
    log.info("to check:  echo .dump %s | sqlite3 %s  " % (tn,dbp) )
    log.info("to check: cat %s | python -m simplejson.tool " % jsp )




def main():
    """
    """
    import sys	
    if len(sys.argv) > 1:
        hub = sys.argv[1]
    else:
	hub = os.environ['NODE_TAG']    

    hubcnf = cnf_(hub)
    fabloop( hubcnf, monitor )


if __name__ == '__main__':
    main()	

