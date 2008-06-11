
#import inspect

import logging
log =  logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def whereami(gbls):
    #print gbls
    print "file:%s name:%s doc:%s " % ( gbls['__file__'], gbls['__name__'], gbls['__doc__'] ) 


#    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52315
#
# obtaining the name of a function/method
# Christian Tismer
# March 2001

# the following function might be implemented in the
# sys module, when generators are introduced.
# For now, let's emulate it. The way is quite inefficient
# since it uses an exception, but it works.

import sys

def _getframe(level=0):
    try:
        1/0
    except:
        import sys
        tb = sys.exc_info()[-1]
    frame = tb.tb_frame
    while level >= 0:
        frame = frame.f_back
        level = level - 1
    return frame

#sys._getframe = _getframe
#del _getframe

# we now assume that we have sys._getframe

def funcname():
    return _getframe(1).f_code.co_name

def ctx(g):
    
    name = _getframe(1).f_code.co_name
    
    #print "ctx... %s " % name 
    
    c = None
    
    if isinstance( g , dict ):   ## globals
        try:
            f = g[name]
        except KeyError, ke:
            #print "keyerror for %s " % g 
            f = None
    elif isinstance( g , object ):
        c = g.__class__
        f = c.__dict__[name]
    else:
        f = None
     
    if f:
        m = f.__module__
    else:
        m = None
     
    pos = []
    
    if m:
        pos.append(m)
    else:
        pos.append("")
        
    if c:
        pos.append(c.__name__)
    else:
        pos.append("")
        
    if name:
        pos.append(name)
    else:
        pos.append("")
        
    return "/".join(pos)

def present(place):
    print place
    #log.debug(place)



