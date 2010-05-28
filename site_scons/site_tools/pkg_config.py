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
    portbin = '/opt/local/bin'
    if env.Bit('mac') and os.path.isdir(portbin): 
        env.PrependENVPath('PATH', portbin )   
    else:
        pass

    if env.WhereIs('pkg-config') is None:
        print "could not find pkg-config : failed access config info for  %s " % pkg 
        env.Exit(1) 

    print "pkg_config for pkg %s t %s " % ( pkg , t )  
    if pkg:
        if not(t):
            t = pkg
        env['PKG_%s' % t.upper() ] = pkg
        env.ParseConfig("pkg-config %s --cflags --libs" % pkg )


