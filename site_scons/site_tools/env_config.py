import os

def generate(env, pkg=None, tool=None ):
    """
       Adjust PATH to pick up the env-config for env managed packages
       Hmm this could benefit from its __file__ to avoid need for envvar ENV_HOME ??
    """
    name = 'ENV_HOME'
    home = os.environ.get(name,None)
    if home and os.path.isdir(home): 
        env.PrependENVPath('PATH', home + os.sep + 'bin' )   
    else:
        print "envvar %s is not defined " % name
        env.exit(1)

    if not(pkg):
        return

    if pkg in ('cjsn','priv',):
        print "env_config : special handling for SCT/SCons built pkg : %s  " % pkg
        env.Append( 
           CPPPATH=['$INCLUDE_ROOT'] ,
           LIBS=[pkg] ,
        ) 
    else: 
        print "env_config : for pkg %s " % pkg  
        if not(tool):
            tool = pkg
        env['PKG_ENV_%s' % tool.upper() ] = pkg
        env.ParseConfig("env-config %s --cflags --libs" % pkg )
    pass 
