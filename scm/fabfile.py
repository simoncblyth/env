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

import os, platform, pwd
#import fabric_patches
from fabric.api import run, env, abort
from fabric.state import output
from fabric.exceptions import NetworkError

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from scm_monitor import GZCheck
from pprint import pformat

output.stdout = False
env.use_ssh_config = True
env.timeout = 3
env.abort_on_prompts = True
env.skip_bad_hosts = True

env.roledefs = {
   'C2':"C H1".split(),
  'C2R':"C H1".split(),
   'G':"Z9:229".split(),
   'T2':"C H1 N".split(),
}

role2srvnode = dict(C2="cms02",G="g4pb")

def target_localize():
    if env.host_string in ("Z9","A","Z9:229"):
        env.shell = '/opt/bin/bash -l -c'	
    else:
        env.shell = '/bin/bash -l -c'	
    #log.info("env %s " % pformat(env) )
    log.info("for env.host_string %s using shell %s " % ( env.host_string, env.shell ))

def scm_sshcheck():
    """
    http://stackoverflow.com/questions/1956777/how-to-make-fabric-ignore-offline-hosts-in-the-env-hosts-list
    """
    target_localize()   
    name = None
    try:
        name = run("uname -n")
    except NetworkError, ne:
        log.warn("NetworkError against %s : %s" % ( env.host_string, ne ))
    except Exception, ex:
        raise ex

    log.info("env.host_string %s name %s " % ( env.host_string, name ) )


def scm_backup_monitor():
    """
    Note that fabric ``run`` returns multiline strings delimited by windows newline ? CR+LF   

    Node splitting 
    """
    target_localize()

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

