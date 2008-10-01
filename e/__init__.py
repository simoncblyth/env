"""
  This function gives access to the bash functions from python :
     from e import bash as e_
     dir = e_(["apache-", "apache-logdir"]).strip()  

"""

import os 
def bash(args):
    return os.popen("bash -c \" . $ENV_HOME/env.bash ; env- ; %s \" " % ";".join(args)).read()
