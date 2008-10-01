"""
  This function gives access to the "env" repository bash functions from python :
    
     import e
     dir = e.bash("apache-", "apache-logdir") 
     dir = e.bash("apache- apache-logdir")
     
  For more control use :   
     fd = e.bash_("apache- apache-logdir")

"""

env = ". $ENV_HOME/env.bash ; env- "

def bash_(*argv):
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
    return os.popen("bash -c \" %s ; %s \" " % ( env , ";".join(args))  )


def bash(*argv):
    return bash_(*argv).read().strip()


if __name__=='__main__':
    dir1 = bash("apache- apache-logdir")
    dir2 = bash(["apache-", "apache-logdir"])
    dir3 = bash("apache-","apache-logdir")
    
    assert dir1 == dir2 == dir3 
    assert dir1 != ""
    print dir1 