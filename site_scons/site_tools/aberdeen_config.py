import os

def generate(env, pkg=None ):
    """
       Adjust PATH to pick up the aberdeen-config for aberdeen managed packages
       
       If this was living inside aberdeen repo it
       could benefit from its __file__ to avoid need for envvar ABERDEEN_HOME ??
       
    """
    name = 'ABERDEEN_HOME'
    home = os.environ.get(name,None)
    if home and os.path.isdir(home): 
        env.PrependENVPath('PATH', home + os.sep + 'bin' )   
    else:
        print "envvar %s is not defined " % name
        env.exit(1)

    print "aberdeen_config for pkg %s " % pkg  
    if pkg:
        env['PKG_ABERDEEN_%s' % pkg.upper() ] = pkg
        env.ParseConfig("aberdeen-config %s --cflags --libs" % pkg )


