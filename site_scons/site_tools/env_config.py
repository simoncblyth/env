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

    if not(pkg):
        return

    ## special handling for local pkgs ... hmm the libname ... 
    if pkg in ('cjsn'):
        env.Append( 
           CPPPATH=['$INCLUDE_ROOT'] ,
           LIBS=[pkg],
        ) 
    else: 
        if not(tool):
            tool = pkg
        env['PKG_ENV_%s' % tool.upper() ] = pkg
        env.ParseConfig("env-config %s --cflags --libs" % pkg )

         



