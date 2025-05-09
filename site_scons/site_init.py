"""  
   Following the SCT site_scons/site_init.py example of how to add global methods 
   Prefixing added global methods with "E" as hint at origin ..
"""
import __builtin__
import usage_log

def ESortedDump(env):
    print "\n".join(["%s:%s"% _ for _ in sorted(env.Dictionary().items())])

def EIncludes(env, sources ):
    """
       INCLUDE_BASE needs to be defined in the main.scons
    """
    env.Replicate( "$INCLUDE_BASE" , sources )

def EWorkaroundSwallowedXlinker(env):
    """
        workaround SCT/SCons bug whereby a repeated -Xlinker flag 
        is swallowed see env ticket #275
    """
    if env.Bit("mac"):
        env.Prepend( LINKFLAGS=['-Xlinker'] )


def ESiteInitMain():
    # Bail out if we've been here before. This is needed to handle the case where
    # this site_init.py has been dropped into a project directory.
    if hasattr(__builtin__, 'ESortedDump'):
        return
    usage_log.log.AddEntry('env site init')
    __builtin__.ESortedDump = ESortedDump
    __builtin__.EIncludes   = EIncludes
    __builtin__.EWorkaroundSwallowedXlinker = EWorkaroundSwallowedXlinker
    pass



ESiteInitMain()

