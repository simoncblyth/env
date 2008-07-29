#!/usr/bin/env python

"""
    PyAlgorithms that assert cause the Gaudi eventloop to 
    terminate but the assert does not surface  .. presenting a problem
    for the test runner, as it needs to record the outcome and the traceback
   
      Whats happening
           AppMgr().run(n) ...
        
           gaudi/GaudiKernel/src/Lib/MinimalEventLoopMgr.cpp 
      invokes the C++ wrapper for the py alg      
           gaudi/GaudiPython/GaudiPython/Algorithm.h
      invoking  call_python_method from :
           gaudi/GaudiPython/src/Lib/CallbackStreamBuf.cpp
   
   
   How to overcome ?
      a) implement a singleton testrunner with 
         an assert method that outputs the stack trace  
      b) signal Gaudi to exit 
   
    these will work but will not solve the issue ...
    
    The nose api passes along an "err" tuple that responds to   
         traceback.format_exception(*err)  
         traceback.format_exception(etype, value, tb, limit=None) 
              
   PROBLEM: how to propagate FAIL/ERROR/SUCCESS status to nose
    
In [15]: traceback.print_exception( AssertionError , "hello" , None )
AssertionError: hello        
            
        
   The "enveloping" nose test runner is in "run" method at :
      /usr/local/env/trac/package/nosenose/0.10.3-release/nose/case.py +113      
                       
    which stores the err if the exception occurs  :
          err = sys.exc_info()   
          result.addError(self, err)                       
                     
                                          
                                                                

"""

import sys
import traceback
err = None

def test_bare_assert():
    assert False
   
def test_caught_assert():
    try:
        assert False
    except AssertionError, ae:
        print "except ae %s" % ae
    finally:
        print "finally"     
       
def test_caught_and_raised_assert():
    try:
        assert False
    except AssertionError, ae:
        print "except ae %s" % ae
        raise ae
    finally:
        print "finally"

def test_caught_recorded_and_reported():
    global err 
    err = None
    try:
        assert False
    except:
        err = sys.exc_info() 
    finally:
        if err:
            traceback.print_exception( *err )