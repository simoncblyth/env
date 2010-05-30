"""
   Convert the SCT/SCons $LIBPATH list  into a (DY)LD_LIBRARY_PATH

"""
def generate(env):
    path = ":".join([env.subst("$LIB_DIR")] + env.SubstList2('$LIBPATH')) 
    env.PrependENVPath( ('','DY')[env.Bit('mac')] + 'LD_LIBRARY_PATH' , path )





