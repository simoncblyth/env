"""
  Not worked on yet ... and not working 
  
  would speed up bash function access to keep the subprocess around
  talk to it as needed...
  
  perhaps easier to use pexpect 

"""

class Bash:
    def __init__(self, env=None):
        import subprocess
        self.prc = subprocess.Popen(['/bin/bash'] , stdout=subprocess.PIPE , stderr=subprocess.PIPE  )
        if env:
            self(env)
        
    def __call__(self,cmd):
        return self.prc.communicate(input=cmd)
        
        
if __name__=='__main__':
    #b = Bash('. $ENV_HOME/env.bash')
    #b("env-usage")
    b = Bash()
    o,e = b("echo hello")