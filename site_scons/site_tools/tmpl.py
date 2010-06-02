
from _tmpl import Tmpl

from SCons.Builder import Builder
from SCons.Action import Action
from SCons import Node, Util

def tmpl_env( env ):
    d = {}
    for k in env['TMPL']:
        d[k] = env.subst('$'+k)   
    print "tmpl_env : %s --> %s " % ( repr(env['TMPL']), repr(d) )
    return d

def tmpl_fill_(target, source, env):
    text = open(str(source[0]), 'r').read()
    tmpl = Tmpl(text)
    out = open(str(target[0]), 'w+')
    d = tmpl_env(env)
    print "tmpl_fill_ %s " % repr(d)
    out.write(tmpl.safe_substitute( d ))
    
def tmpl_emitter(target, source, env):
    """
       refill the template when expanded dict changes
    """
    env.Depends(target, Node.Python.Value( tmpl_env(env) ))
    return (target, source)

def generate(env, **kw):
    """
       The TMPL dict specifies what to propagate into the template
    """
    #if env.get('TMPL', None):
    #   env['TMPL'].update(**kw)
    #else:
    #   env['TMPL'] = kw 

    tmpl_action = Action( tmpl_fill_ , "Fill template $SOURCE creating $TARGET " )
    filltmpl_builder = Builder(action = tmpl_action ,
	                       emitter = tmpl_emitter,               
	                    src_suffix = '.in',
	                  single_source = True )

    env['BUILDERS']['Filltmpl'] = filltmpl_builder


