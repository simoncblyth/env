"""
   Convert the SCT/SCons $LIBPATH list  into a (DY)LD_LIBRARY_PATH
     path = ":".join( [env.subst("$LIB_DIR")] + env.SubstList2('$LIBPATH')) 




"""
def generate(env, dirs=[]):
    path = ":".join( [ env.subst(x) for x in dirs ]) 
    print "libpath set to : %s " % path 
    env.PrependENVPath( ('','DY')[env.Bit('mac')] + 'LD_LIBRARY_PATH' , path )





