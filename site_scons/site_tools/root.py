#
# Based on 
#   http://www.scons.org/wiki/GCCXMLBuilder 
#   http://www.scons.org/wiki/UnTarBuilder
#
#  Without taking precarious measures like imbibing the callers PATH 
#  this has to depend on ROOTSYS being defined in order to locate rootcint and root-config
#
#  It could be avoided by spoofing a root.pc ... but thats too much effort to maintain
#

import os
import SCons.Builder
import SCons.Tool


ROOTCINTBuilder = SCons.Builder.Builder(action = "$ROOTCINT -f $TARGET -c $_CPPINCFLAGS $SOURCES ",
                              suffix='_Dict.cxx',
                              src_suffic = ['h', 'hpp'],
                              source_scanner = SCons.Tool.CScanner)

def generate(env):
    rootsys = os.environ.get('ROOTSYS', None)
    if rootsys is None:
        print 'ERROR : %s : ROOTSYS is not defined ' % __file__
        env.Exit(1)
    
    env.PrependENVPath('PATH', rootsys + os.sep + 'bin' )

    if env.WhereIs('root-config') is None:
        print "ERROR : %s :  no root-config in PATH " % __file__
        env.Exit(1)
    env.ParseConfig("root-config --cflags --glibs")
   
    #if env.Bit("linux"): 
    #    env.PrependENVPath('LD_LIBRARY_PATH', rootsys + os.sep + 'lib' )

    rootcint_path = env.WhereIs('rootcint')
    if rootcint_path is None:
        print 'Could not find rootcint, please make sure it is on your PATH'
        env.Exit(1)

    env['ROOTCINT'] = rootcint_path
    rootcint_dir = os.path.dirname(rootcint_path)
    env.Append(BUILDERS = {'RootcintDictionary' : ROOTCINTBuilder })

def exists(env):
    return env.Detect('root')






