def generate(env, **kw ):
    d = env.Dictionary()
    print "\n".join( ["%s:%s" % (k,d.get(k,None)) for k in sorted(kw)])
        





