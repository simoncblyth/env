import os

def exists(env):
    return True

def generate(env):
    """
       Adjust PATH to pick up the pkg-config for port installed packages
    """
    portbin = '/opt/local/bin'
    if env.Bit('mac') and os.path.isdir(portbin): 
        env.PrependENVPath('PATH', portbin )   
    else:
        pass


