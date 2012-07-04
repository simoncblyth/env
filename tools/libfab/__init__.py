#!/usr/bin/env python
"""
Libary Usage of Fabric
========================

This module attempts to 

#. avoid repetition in Fabric usage 
#. facilitate usage of fabric as a library
#. break from the shackles of the fabfile

   #. http://docs.fabfile.org/en/1.4.2/usage/library.html


Usage, to run a task against a list of remote hosts and collect returns:: 

	from fabric.api import run
	from env.tools.libfab import fabloop

	def check(host):
	    return run("uname -a")

        if __name__ == '__main__':
	    ret = fabloop('C C2 H N N1', check)
            print ret

"""
import logging, os, pwd
log = logging.getLogger(__name__)
from fabric.api import env, run
from fabric.state import output
from fabric.network import disconnect_all
from fabric.exceptions import NetworkError
from pprint import pformat

def check(host):
    """
    Check remote host is alive by running uname command on it 
    """
    name = None
    try:
        name = run("uname -n")
    except NetworkError, ne:
        log.warn("NetworkError against %s : %s" % ( env.host_string, ne ))
    except Exception, ex:
        raise ex

    return name
    log.info("env.host_string %s name %s " % ( env.host_string, name ) )


def localize(host):
    """
    Remote host specific localization, including setting the host_string

    :param host: host_string

    TODO:
     
    #. Perhost settings such as the shell to use, should be defined in config rather than in code  
    """
    env.host_string = host
    if host in ("Z9","A","Z9:229"):
        env.shell = '/opt/bin/bash -l -c'	
    else:
        env.shell = '/bin/bash -l -c'	
    #log.info("env %s " % pformat(env) )
    log.info("for env.host_string %s using shell %s " % ( env.host_string, env.shell ))

def setup(hosts):
    """


    # print output
    # {'status': True, 'warnings': True, 'stdout': True, 'running': True, 'user': True, 'stderr': True, 'aborts': True, 'debug': False}

    """
    output.stdout = False
    output.running = False
    logging.basicConfig(level=logging.WARN)
    env.use_ssh_config = True
    env.abort_on_prompts = True
    env.skip_bad_hosts = True
    env.timeout = 3
    env.hosts = hosts.split()

def fabloop( hosts , task=lambda _:_):
    """
    :param hosts: space delimited string containg host ssh tags
    :param task: fabric function to be performed on each host
    :return: dict keyed by host containing the return from teh task
    """
    setup(hosts)
    ret = {}
    for host in env.hosts:
        localize(host)
        ret[host] = task(host)
    teardown()	
    return ret

def teardown():
    disconnect_all()



def file_exists(path):
    """Tests if there is a *remote* file at the given location."""
    return run('test -e "%s" && echo OK ; true' % (path)).endswith("OK")

def file_attribs_get(path):
    """Return mode, owner, and group for remote path.
       Return mode, owner, and group if remote path exists, 'None'
       otherwise.
    """
    if file_exists(path):
        fs_check = run('stat %s %s' % (path, '--format="%a %U %G"'))
        (mode, owner, group) = fs_check.split(' ')
	return {'mode': mode, 'owner': owner, 'group': group}
    else:
	return None


def user():
    user = pwd.getpwuid(os.getuid())[0]
    return user
 

if __name__ == '__main__':
    ret = fabloop('C C2 N N1 H',check)
    print pformat(ret)
    
