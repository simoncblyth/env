#
#   http://www.scons.org/wiki/GCCXMLBuilder 
#
# Copyright 2007 Joseph Lisee
#
# Author: Joseph Lisee
# License: Public Domain

"""
This module contains a custom GCC-XML tool for SCons
"""

import os

import SCons.Builder
import SCons.Tool



GCCXMLBuilder = SCons.Builder.Builder(action = "$GCCXML $GCCXML_EXTRA_FLAGS $_XML_CPPINCFLAGS $_XML_CPPDEFFLAGS $SOURCE -fxml=$TARGET",
                              suffix='xml',
                              src_suffic = ['h', 'hpp'],
                              source_scanner = SCons.Tool.CScanner)

def generate(env):
    gccxml_path = env.WhereIs('gccxml')
    if gccxml_path is None:
        print 'Could not find gccxml, please make sure it is on your PATH'
        env.Exit(1)

    env['GCCXML'] = gccxml_path

    gccxml_dir = os.path.dirname(gccxml_path)
    extra = ''
    if os.name != 'posix':
        extra = '--gccxml-config "' + os.path.abspath(os.path.join(gccxml_dir, 'gccxml_config')) +'"'
        extra += ' --gccxml-cxxflags " /DWIN32 /D_WINDOWS /W3 /Zm1000 /EHsc /GR /MT" '
    env['GCCXML_EXTRA_FLAGS'] = extra
    #env['GCCXML_EXTRA_FLAGS'] = ''

    # These variables hold the expanded form of the include and defines lists
    env['_XML_CPPINCFLAGS'] = '$( ${_concat(INCPREFIX, XMLCPPPATH, INCSUFFIX, __env__, RDirs)} $)'
    env['_XML_CPPDFFFLAGS'] = '${_defines(CPPDEFPREFIX, XMLCPPDEFINES, CPPDEFSUFFIX, __env__)}'


    if os.name != 'posix':
        env['GCCXML_INCPREFIX'] = '-I'
        env['_XML_CPPINCFLAGS'] = '$( ${_concat(GCCXML_INCPREFIX, CPPPATH, INCSUFFIX, __env__, RDirs, TARGET, SOURCE) } $)'

    # Added the builder to the given environment
    env.Append(BUILDERS = {'XMLHeader' : GCCXMLBuilder })

def exists(env):
    return env.Detect('gccxml')






