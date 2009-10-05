"""
   Note this implementation, 
   creates a new bash process for every command  
   
   Also .. no error handling : better to use subprocess 
   
   
     from env.bash import Bash
     b = Bash()
     dir = b("apache-", "apache-logdir") 
     dir = b("apache- apache-logdir")
     
  For more control use :   
      pipe = b.cmd("apache- apache-logdir")

"""

class Bash:
    def __init__(self, env=". $ENV_HOME/env.bash ; env-" ):
        self.env = env
    @classmethod
    def exe(cls, *argv):
        b = Bash()
        return b(*argv)
        
    def cmd( self, *argv):
        if len(argv) == 1: 
            if type(argv[0])==str:
                import re
                space = re.compile(r'\s+')
                args = space.split(argv[0].strip())
            elif type(argv[0])==list:
                args = argv[0]
            else:
                raise Exception, "unexpected type"
        else:
            args = argv
        import os        
        return os.popen("bash -c \" %s ; %s \" " % ( self.env , ";".join(args))  )

    def __call__(self, *argv ):
        return self.cmd(*argv).read().strip()


if __name__=='__main__':
    from env.bash import Bash
    b = Bash()
    dir1 = b("apache- apache-logdir")
    dir2 = b(["apache-", "apache-logdir"])
    dir3 = b("apache-","apache-logdir")
    
    assert dir1 == dir2 == dir3 
    assert dir1 != ""
    print dir1
     
