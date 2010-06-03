import os

def generate(env, pkg=None, t=None):
    """
       Adjust PATH to pick up the pkg-config 
       for port installed packages

       And use it to grab the config info for the pkg   

           pkg  : pkg name as used in the pkg-config call, 
                    eg glib-2.0 , libpcre
 
           t    : abbreviated name of the tool, 
                   eg  glib, pcre  
                  which corresponds to the __name__
                     
    """
    
    if env.Bit('mac'):
        portbin = '/opt/local/bin'
        if os.path.isdir(portbin): 
            env.PrependENVPath('PATH', portbin )   
    else:
        pass

    if env.WhereIs('pkg-config') is None:
        print "could not find pkg-config : failed access config info for  %s " % pkg 
        env.Exit(1) 

    envpfx = os.environ.get('ENV_PREFIX',None)
    if not(envpfx):
        print "error ENV_PREFIX is not defined, see : local-usage "
        env.Exit(1)

    pcdir = os.path.join( envpfx , 'lib' , 'pkgconfig' )
    if not(os.path.isdir(pcdir)):
        print "error pkgconfig dir does not exist see : pkgconfig-usage "
        env.Exit(1)

    env.PrependENVPath('PKG_CONFIG_PATH', pcdir )
    print "pkg_config for pkg %s t %s " % ( pkg , t )  
    if pkg:
        if not(t):
            t = pkg
        env['PKG_SYS_%s' % t.upper() ] = pkg
        if env.WhereIs('%s-config' % t ):
            print "using %s-config " % t 
            env.ParseConfig("%s-config --cflags --libs" % t)
        else:          
            env.ParseConfig("pkg-config %s --cflags --libs" % pkg )

