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

    print "env_config for pkg %s " % pkg  
    if pkg:
        if not(tool):
            tool = pkg
        env['ENV_PKG_%s' % tool.upper() ] = pkg
        env.ParseConfig("env-config %s --cflags --libs" % pkg )

         



