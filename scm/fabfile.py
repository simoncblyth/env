#!/usr/bin/env python
"""
Usage::

    fab        scm_backup_monitor    # on the env.hosts nodes
    fab -R C2 scm_backup_monitor     # on nodes with role 'C2'

Typically cron invoked with bash functions::

   scm-backup-monitor
   scm-backup-monitorw

How to run init code ?

#. avoid repeating logging setup for example without doing it at module level



Approach:

#. keep fabfile simple, 

   #. just the shim to do the remote commands 
   #. pass returned vales on to checkers implemented elsewhere    


For bare ipython interactive tests::

    from fabric.api import run, env
    env.use_ssh_config = True
    env.hosts = ["WW"]
    cmd = "find $SCM_FOLD/backup/$LOCAL_NODE -name '*.gz' -exec du --block-size=1M {} \;"
    paths = run(cmd)

Check whats in DB with::

    echo .dump tgzs | sqlite3 $LOCAL_BASE/env/scm/scm_backup_monitor.db

Check and prettyprint the json with::

    cat $APACHE_HTDOCS/data/scm_backup_monitor.json | python -m simplejson.tool


Regards monitoring organization:

#. repo nodes are most appropriate for running the check, 

   #. they have all keys already 
   #. they are webservers and hence can sphinx present the status
 
Curiously:

#. there are windows newlines ``\r\n`` in the returned string not ``\n`` 

"""

import os, platform
from fabric.api import run, env, abort

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from env.tools.libfab import fabloop
from scm_monitor import GZCheck

def monitor(node):
    """
    Note that fabric ``run`` returns multiline strings delimited by windows newline ? CR+LF   

    Workaround for fabric strictures by writing to DB separately for each remote and 
    benefiting from the uniqing (re-runability) build into the SQL. 
    """

    tn    = "tgzs"
    srvnode = "g4pb"   # how to configure this hub specifier g4pb/cms02 
    dbp   = os.path.expandvars("$LOCAL_BASE/env/scm/scm_backup_monitor.db")   
    jsonp = os.path.expandvars("$APACHE_HTDOCS/data/scm_backup_monitor_%(node)s.json" % locals() ) 

    assert os.path.exists(os.path.dirname(jsonp)), jsonp
    if not os.path.exists(os.path.dirname(dbp)):
        os.makedirs(os.path.dirname(dbp))

    ## parse the response and update DB table accordingly 

    gzk = GZCheck(dbp, tn, srvnode)
    ret = run(gzk.cmd)
    gzk(ret.split("\r\n"), node )  
    gzk.check()

    select = ["%s/%s" % ( type, proj) for type in ("tracs","repos","svn") for proj in ("env","heprez","dybaux","dybsvn","tracdev","aberdeen","workflow")]
    gzk.jsondump(jsonp, node=env.host_string, select=select)

    pass
    log.info("to check:  echo .dump %s | sqlite3 %s  " % (tn,dbp) )
    log.info("to check: cat %s | python -m simplejson.tool " % jsonp )


if __name__ == '__main__':
    fabloop( 'Z9:229', monitor )
    #fabloop( 'C H1', monitor )
