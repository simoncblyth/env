
def generate(env):
    name = 'libpcre'
    env['PCRE_EXTERNAL_NAME'] = name
    env.Tool('macports')
    env.ParseConfig("pkg-config %s --cflags --libs" % name )

def exists(env):
    return True






