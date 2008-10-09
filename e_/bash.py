"""
   
     import e_
     dir = e_.bash_("apache-", "apache-logdir") 
     dir = e_.bash_("apache- apache-logdir")
     
  For more control use :   
     fd = e_.bash__("apache- apache-logdir")

"""

env = ". $ENV_HOME/env.bash ; env- "

def bash__(*argv):
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


def bash_(*argv):
    return bash__(*argv).read().strip()


if __name__=='__main__':
    dir1 = bash_("apache- apache-logdir")
    dir2 = bash_(["apache-", "apache-logdir"])
    dir3 = bash_("apache-","apache-logdir")
    
    assert dir1 == dir2 == dir3 
    assert dir1 != ""
    print dir1 