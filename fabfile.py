
from fabric.api import run, env

env.use_ssh_config = True
env.hosts = ["N","N1","C","C2","H"]

def hostname():
    run('uname -n')

