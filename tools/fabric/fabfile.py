from fabric.api import run, env
from fabric.state import output

import re

# print output
# {'status': True, 'warnings': True, 'stdout': True, 'running': True, 'user': True, 'stderr': True, 'aborts': True, 'debug': False}

output.stdout = False

env.timeout = 2
env.skip_bad_hosts = True
env.use_ssh_config = True



env.hosts = ["N","N1","C","C2","H"]
#env.hosts = ["C2","WW"]
#env.hosts = ["C2"]
#env.hosts = ["WW"]

env.roledefs = {
    'svn':['C2','WW'],
    'web':['C2','C1','H'],
}


def file_exists(location):
    """Tests if there is a *remote* file at the given location."""
    return run('test -e "%s" && echo OK ; true' % (location)).endswith("OK")

def file_attribs_get(location):
    """Return mode, owner, and group for remote path.
       Return mode, owner, and group if remote path exists, 'None'
       otherwise.
    """
    if file_exists(location):
        fs_check = run('stat %s %s' % (location, '--format="%a %U %G"'))
        (mode, owner, group) = fs_check.split(' ')
	return {'mode': mode, 'owner': owner, 'group': group}
    else:
	return None


def hostname():
    fp = run('uname -n')
    print "[%s] %s " % ( fp, env.host_string )


