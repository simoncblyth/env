def generate(env):
    print "\n".join( ["%s:%s" % _ for _ in sorted(env['ENV'].items())])
        





