"""
   This mimicks pkg_config.py for internally built libs 
  
   To add a pkg :
      * publish includes to INCLUDE_BASE using the EIncludes global func, eg :
          EIncludes( ec, ['private.h'] )

   NB the "pkg.py" name (eg priv.py) is assumed to match the library name "libpriv"
"""

def generate(env, pkg=None ):
    if pkg in ('cjsn','priv','addroot','envtools'):
        print "env_config : for SCT/SCons built pkg : %s : append to CPPATH and LIBS   " % pkg
        env.Append( 
           CPPPATH=['$INCLUDE_BASE'] ,
           LIBS=[pkg] ,
        ) 
    else: 
        print "env_config : ERROR pkg %s is not handled by env_config ... try pkg_config " % pkg  
    pass 
