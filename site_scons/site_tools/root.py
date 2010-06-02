"""
 Based on 
   http://www.scons.org/wiki/GCCXMLBuilder 
   http://www.scons.org/wiki/UnTarBuilder

  Without taking precarious measures like imbibing the callers PATH 
  this has to depend on ROOTSYS being defined in order to locate rootcint and root-config


  Problems...
       target cleaning doesnt reach all the derived ...

g4pb:e blyth$ find scons-out -type f
scons-out/.sconsign_darwin.dblite
scons-out/dbg/obj/aberdeen/AbtViz/AbtVizDict_Dict.h
scons-out/dbg/obj/aberdeen/DataModel/AbtDataModelDict_Dict.h
scons-out/dbg/obj/rootmq/rootmq_Dict.h

g4pb:e blyth$ find scons-out -type l
scons-out/dbg/lib/libAbtDataModel.so
scons-out/dbg/lib/libAbtViz.so
scons-out/dbg/lib/librootmq.so


   How to use "string type actions" from a python action ? 
      * in order to try to use env.Clean with the rootcint_builder


"""

import os
import SCons.Builder
import SCons.Tool


rootcint_builder = SCons.Builder.Builder(
          action = "$ROOTCINT -f $TARGET -c $_CPPINCFLAGS $SOURCES ",
          suffix = '_Dict.cxx',
      src_suffix = ['h', 'hpp'],
  source_scanner = SCons.Tool.CScanner,
)

def rootsolink_(target, source, env):
    """
        Note that any target passed to the builder is being ignored ?
    """
    if not(env.Bit("mac")):
        return
    suffix = env.subst('$SHLIBSUFFIX') 
    sofix = lambda x:x[0:-len(suffix)] + '.so'
    iwd = os.getcwd()
    for s in source:
        dir = s.dir.abspath
        os.chdir( dir )
        sn = s.name
        if sn.endswith(suffix):
            tn = sofix( sn )
            #env.Clean( s , dir + os.sep + tn ) 
            if not(os.path.exists( tn )):
                os.symlink( sn , tn )         
                print "rootsolink_ %s %s : symlink source to target " % ( tn, sn ) 
            else:
                print "rootsolink_ %s %s : target exists already " % ( tn, sn ) 
        else:
            print "rootsolink_ skip as name %s does not end with expected suffix %s " % ( sn , suffix ) 
    os.chdir( iwd )
    pass

rootsolink_builder = SCons.Builder.Builder( 
    action=rootsolink_ , 
     suffix = '.so',
 src_suffix = ['.dylib'],
    source_factory=SCons.Node.FS.default_fs.Entry,
    target_factory=SCons.Node.FS.default_fs.Entry,
)

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

    env['ROOT_LIBDIR'] = rootsys + os.sep + 'lib'
    env['ROOTCINT'] = rootcint_path

    env.Append( BUILDERS = {
         'RootcintDictionary' : rootcint_builder ,
                 'Rootsolink' : rootsolink_builder ,
       })

def exists(env):
    return env.Detect('root')






