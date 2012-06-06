
from fabric.api import run, env

env.timeout = 2
env.skip_bad_hosts = True
env.use_ssh_config = True
env.hosts = ["N","N1","C","C2","H"]

def hostname():
    fp = run('uname -n')
    print "[%s]" % fp


