"""
Usage::

    fab        scm_backup_monitor    # on the env.hosts nodes
    fab -R svn scm_backup_monitor   # on nodes with role 'svn'


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

import os
from fabric.api import run, env, abort
from fabric.state import output
import logging
log = logging.getLogger(__name__)

from scm_monitor import GZCheck

output.stdout = False
env.use_ssh_config = True

#env.hosts = ["WW"]
env.hosts = ["C", "H1"]
#env.hosts = ["H1"]


def scm_backup_monitor():
    """
    Note that fabric ``run`` returns multiline strings delimited by windows newline ? CR+LF   


    Node splitting 
    """
    logging.basicConfig(level=logging.INFO)

    node = env.host_string
    tn = "tgzs"
    dbp   = os.path.expandvars("$LOCAL_BASE/env/scm/scm_backup_monitor.db")   # into SCM_FOLD maybe ? /var/scm
    jsonp = os.path.expandvars("$APACHE_HTDOCS/data/scm_backup_monitor_%(node)s.json" % locals() ) 

    assert os.path.exists(os.path.dirname(jsonp)), jsonp
    if not os.path.exists(os.path.dirname(dbp)):
        os.makedirs(os.path.dirname(dbp))

    gzk = GZCheck(dbp, tn)
    ret = run(gzk.cmd)
    gzk(ret.split("\r\n"), node )  
    
    gzk.check()

    select = ["%s/%s" % ( type, proj) for type in ("tracs","repos","svn") for proj in ("env","heprez","dybaux","dybsvn","tracdev","aberdeen",)]
    gzk.jsondump(jsonp, node=env.host_string, select=select)

    pass

    log.info("to check:  echo .dump %s | sqlite3 %s  " % (tn,dbp) )
    log.info("to check: cat %s | python -m simplejson.tool " % jsonp )

