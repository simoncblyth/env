"""
Usage::

    fab        scm_backup_check    # on the env.hosts nodes
    fab -R svn scm_backup_check    # on nodes with role 'svn'


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

    echo .dump tgzs | sqlite3 scm_backup_check.db 

Regards monitoring organization:

#. repo nodes are most appropriate for running the check, 

   #. they have all keys already 
   #. they are webservers and hence can sphinx present the status
 

"""

from fabric.api import run, env, abort
from fabric.state import output
import logging
log = logging.getLogger(__name__)

from scm_monitor import GZCheck

output.stdout = False
env.use_ssh_config = True

#env.hosts = ["WW"]
env.hosts = ["C"]


def scm_backup_check():
    """
    """
    logging.basicConfig(level=logging.INFO)
    gzk = GZCheck("scm_backup_check.db")
    ret = run(gzk.cmd)
    assert ret.return_code == 0, "non zero rc %s from %s " % ( ret.return_code, gzk.cmd )
    #print ret.split("\r\n")
    gzk(ret.split("\r\n"), env.host_string )   # why the windows newline ? CR+LF   
    gzk.check()
