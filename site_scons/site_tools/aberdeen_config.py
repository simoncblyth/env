import os

def generate(env, pkg=None ):
    """
       Adjust PATH to pick up the aberdeen-config for aberdeen managed packages
       
       If this was living inside aberdeen repo it
       could benefit from its __file__ to avoid need for envvar ABERDEEN_HOME ??
      
       This is still using the fragile bash function approach ... requiring
               aberdeen-
               abtmodel- 
 
    """
    name = 'ABERDEEN_HOME'
    home = os.environ.get(name,None)
    if home and os.path.isdir(home): 
        bin = home + os.sep + 'bin'   
        print "preprnding %s " % bin   
        env.PrependENVPath('PATH', bin )   
    else:
        print "envvar %s is not defined " % name
        env.exit(1)

    print "aberdeen_config for pkg %s with home %s  " % ( pkg , home )  
    if pkg:
        env['PKG_ABERDEEN_%s' % pkg.upper() ] = pkg
        cmd = "aberdeen-config %s --cflags --libs" % pkg 
        print "invoking \"%s\" " % cmd 
        env.ParseConfig(cmd )


