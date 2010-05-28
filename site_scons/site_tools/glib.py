
def generate(env):
    name = 'glib-2.0'
    env['GLIB_EXTERNAL_NAME'] = name
    env.Tool('macports')
    env.ParseConfig("pkg-config %s --cflags --libs" % name )

def exists(env):
    return True






