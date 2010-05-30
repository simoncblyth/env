def generate(env):
    env.PrependENVPath( ('','DY')[env.Bit('mac')] + 'LD_LIBRARY_PATH' , env.subst('$LIB_DIR') )





