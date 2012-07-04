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
	    ret = fabloop('G', check)
            print ret


TODO:

#. use bullet proof approach to unreachable targets standardly


interactive testing of Fabric
-------------------------------

For bare ipython interactive tests::

    from fabric.api import run, env
    env.use_ssh_config = True
    env.hosts = ["WW"]
    cmd = "find $SCM_FOLD/backup/$LOCAL_NODE -name '*.gz' -exec du --block-size=1M {} \;"
    paths = run(cmd)


observations
--------------

#. there are windows newlines ``\r\n`` in the Fabric run returned string not ``\n`` 

#. performance benefits hugely by minimizing remote connections by

    #. doing a single remote find to get the paths and sizes   
    #. pulling datetime info encoded into path rather than querying remote file system.


"""
import logging, os, pwd
from ConfigParser import RawConfigParser
log = logging.getLogger(__name__)
from fabric.api import env, run
from fabric.state import output
from fabric.network import disconnect_all
from fabric.exceptions import NetworkError
from pprint import pformat



def rrun(cmd):
    """
    Robust fabric run, which simply returns None when target host not
    reachable without erroring out.

    :param cmd: 
    :return: command response on remote node or None if not reachable
    """
    ret = None
    try:
        ret = run(cmd)
    except NetworkError, ne:
        log.warn("NetworkError against %s : %s" % ( env.host_string, ne ))
    except Exception, ex:
        raise ex
    return ret

def localize(host):
    """
    Remote host specific localization, including setting the host_string

    :param host: host_string

    """
    cnf = getattr( env, 'libfabcnf', None )
    if cnf and cnf.has_section( host ):
	for key in cnf.options(host):    
            val = cnf.get(host,key)
            setattr(env, key, val )
            log.warn("for host %s setting (key,val)  (%s,%s) " % ( host,key,val ))
	    pass
        pass
    env.host_string = host
    log.info("for env.host_string %s using shell %s " % ( env.host_string, env.shell ))

def setup(khosts):
    """
    :param khosts: hosts key pointing to key in HOSTS section or hosts string

    Read ini format config file ~/.libfab.cnf and sets global defaults 

    Example ini file::


            [HOSTS]
	    G = Z9:229
            C2 = C H1  

	    [ENV]
	    verbose = True
	    timeout = 3

	    [Z9:229]
	    shell = /opt/bin/bash -l -c

    # print output
    # {'status': True, 'warnings': True, 'stdout': True, 'running': True, 'user': True, 'stderr': True, 'aborts': True, 'debug': False}
    """

    RawConfigParser.optionxform = str    # case sensitive keys 
    cnf = RawConfigParser()
    path = os.path.expanduser("~/.libfab.cnf")
    cnf.read(path)
    env.libfabcnf = cnf    # non-standard tack on 

    env.use_ssh_config = True
    env.abort_on_prompts = True
    env.skip_bad_hosts = True
    env.timeout = 3

    logging.basicConfig(level=logging.WARN)

    dhosts = {}
    if cnf and cnf.has_section('HOSTS'):
        for key in cnf.options('HOSTS'):    
	    dhosts[key] = cnf.get('HOSTS',key)

    print dhosts

    hosts = dhosts.get(khosts, khosts)    
    env.hosts = hosts.split()
    log.warn("khosts %s env.hosts %s " % (khosts, repr(env.hosts)) )

    if cnf and cnf.has_section('ENV'):
        for key in cnf.options('ENV'):    
            val = cnf.get('ENV',key)
            setattr(env, key, val )
            log.warn("ENV setting (key,val)  (%s,%s) " % ( key,val ))
	    pass

    verbose = getattr(env, 'verbose', False)
    output.stdout = verbose
    output.running = verbose
    output.status = verbose



def fabloop( khosts , task=lambda _:_):
    """
    :param khosts: space delimited string containg host ssh tags or key to a list of groups 
    :param task: fabric function to be performed on each host
    :return: dict keyed by host containing the return from teh task
    """
    setup(khosts)
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
 
def check(host):
    assert host == env.host_string, (host, env.host_string)
    return rrun("uname -n")


if __name__ == '__main__':
    ret = fabloop('G',check)
    print pformat(ret)
    
