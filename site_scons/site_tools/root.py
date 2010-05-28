#
# Based on 
#   http://www.scons.org/wiki/GCCXMLBuilder 
#   http://www.scons.org/wiki/UnTarBuilder
#
#  Want to avoid binding operation to directory layout ...
#     dict_ = env.RootcintDictionary('AbtViz', Glob('*.h'))
#
#

import os
import SCons.Builder
import SCons.Tool


ROOTCINTBuilder = SCons.Builder.Builder(action = "$ROOTCINT -f $TARGET -c $INCLUDES $SOURCES src/LinkDef.hh",
                              suffix='_Dict.cxx',
                              src_suffic = ['h', 'hpp'],
                              source_scanner = SCons.Tool.CScanner)

def generate(env):
    """
          Is it appropriate to access ROOTSYS and do the ParseConfig here, 
          inside the tool ?

    """
    env.Dump()
    #print "tool:root\n" + "\n".join( ["%s:%s" % _ for _ in sorted(os.environ.items()) ])
    rootsys = os.environ.get('ROOTSYS', None)
    if rootsys is None:
        print 'Envvar ROOTSYS is not defined '
        env.Exit(1)
    
    env.PrependENVPath('PATH', rootsys + os.sep + 'bin' )
    env.ParseConfig("root-config --cflags --glibs")

    rootcint_path = env.WhereIs('rootcint')
    if rootcint_path is None:
        print 'Could not find rootcint, please make sure it is on your PATH'
        env.Exit(1)

    env['ROOTCINT'] = rootcint_path
    rootcint_dir = os.path.dirname(rootcint_path)
    env['ROOTCINT_LINKDEF'] = 'LinkDef.hh'
    env.Append(BUILDERS = {'RootcintDictionary' : ROOTCINTBuilder })

def exists(env):
    return env.Detect('root')






