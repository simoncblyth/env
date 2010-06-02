def generate(env, pkg=None ):
    if pkg in ('cjsn','priv','addroot'):
        print "env_config : for SCT/SCons built pkg : %s : append to CPPATH and LIBS   " % pkg
        env.Append( 
           CPPPATH=['$INCLUDE_ROOT'] ,
           LIBS=[pkg] ,
        ) 
    else: 
        print "env_config : ERROR pkg %s is not handled by env_config ... try pkg_config " % pkg  
    pass 
