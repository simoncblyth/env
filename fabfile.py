"""
Usage::

    fab        scm_backup_check    # on the env.hosts nodes
    fab -R svn scm_backup_check    # on nodes with role 'svn'


"""
from fabric.api import run, env
from fabric.state import output

# print output
# {'status': True, 'warnings': True, 'stdout': True, 'running': True, 'user': True, 'stderr': True, 'aborts': True, 'debug': False}

output.stdout = False

env.timeout = 2
env.skip_bad_hosts = True
env.use_ssh_config = True

#env.hosts = ["N","N1","C","C2","H"]
#env.hosts = ["C2","WW"]
env.hosts = ["C2"]


env.roledefs = {
    'svn':['C2','WW'],
    'web':['C2','C1','H'],
}


def hostname():
    fp = run('uname -n')
    print "[%s]" % fp

def scm_backup_check():
    out = run("find $SCM_FOLD/backup/$LOCAL_NODE -name '*.gz' ")
    assert out.return_code == 0, out.return_code
    print "[%s]" % out


