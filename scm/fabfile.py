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

import os, platform, pwd
#import fabric_patches
from fabric.api import run, env, abort
from fabric.state import output
import logging
log = logging.getLogger(__name__)

from scm_monitor import GZCheck

from pprint import pformat

output.stdout = False
env.use_ssh_config = True

env.roledefs = {
   'C2':"C H1".split(),
   'G':"Z9:229".split(),
}

role2srvnode = dict(C2="cms02",G="g4pb")


def scm_backup_monitor():
    """
    Note that fabric ``run`` returns multiline strings delimited by windows newline ? CR+LF   


    Node splitting 
    """
    logging.basicConfig(level=logging.INFO)

    log.info("env.host_string %s " % env.host_string )
    #log.info("env %s " % pformat(env) )

    node = env.host_string
    roles = env.roles
    tn = "tgzs"
    dbp   = os.path.expandvars("$LOCAL_BASE/env/scm/scm_backup_monitor.db")   # into SCM_FOLD maybe ? /var/scm
    jsonp = os.path.expandvars("$APACHE_HTDOCS/data/scm_backup_monitor_%(node)s.json" % locals() ) 

    assert os.path.exists(os.path.dirname(jsonp)), jsonp
    if not os.path.exists(os.path.dirname(dbp)):
        os.makedirs(os.path.dirname(dbp))


    ## intended to normally run from the hub nodes(where repositories are), but for G often testing remote hubs
    localnode = platform.node()
    srvnode = localnode.split(".")[0]

    assert len(roles) == 1 , "expecting one role %s " % repr(roles)
    role = roles[0]
    srvnode = role2srvnode[role]
    if srvnode == 'cms02':
        user = pwd.getpwuid(os.getuid())[0]
        assert user == 'root' , "expect this to be run by root, not '%s' " % user 
    else:
	pass    

    gzk = GZCheck(dbp, tn, srvnode)
    ret = run(gzk.cmd)
    gzk(ret.split("\r\n"), node )  
    
    gzk.check()

    select = ["%s/%s" % ( type, proj) for type in ("tracs","repos","svn") for proj in ("env","heprez","dybaux","dybsvn","tracdev","aberdeen","workflow")]
    gzk.jsondump(jsonp, node=env.host_string, select=select)

    pass

    log.info("to check:  echo .dump %s | sqlite3 %s  " % (tn,dbp) )
    log.info("to check: cat %s | python -m simplejson.tool " % jsonp )

