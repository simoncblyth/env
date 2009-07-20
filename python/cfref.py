#!/usr/bin/env python 
"""

   Attempts to tame the MessageSvc
   based on : gaudi/GaudiPython/tests/test_basics.py::test11ostreamcallback
   
   BUILD_PATH : gaudi/GaudiPython 


"""
import sys
import os

class Collect:
    def __init__(self):
        from cStringIO import StringIO
        self.buf = StringIO()
    def write(self, s):
        self(s)
    def __call__(self, s):
        self.buf.write(s)
    def as_list(self):
        return ["%s\n" % l for l in str(self).split("\n")] 
    def __repr__(self):
        return self.buf.getvalue()


class CaptureMsv(list):
    def __init__(self, collect=None ):
        if not(collect):
            collect = Collect()
        self.collect = collect

        import GaudiPython
        csb  = GaudiPython.CallbackStreamBuf(collect)
        ost = GaudiPython.gbl.basic_ostream('char','char_traits<char>')(csb)
        app = GaudiPython.AppMgr(outputlevel=5)
        msv = app.service('MessageSvc', GaudiPython.gbl.IMessageSvc)
        self.original = msv.defaultStream()
        msv.setDefaultStream(ost)

    def __call__(self):
        import GaudiPython
        app = GaudiPython.AppMgr(outputlevel=5)
        msv = app.service('MessageSvc', GaudiPython.gbl.IMessageSvc)
        msv.setDefaultStream(self.original) 
        for l in self.collect.as_list():
            self.append(l)
    
    def __repr__(self):
        return "".join(["%-4d %s" % (n,l)  for n,l in enumerate(self)] )

        
def test_msv_capture():
    """
          This fails ...
          but test_msv_base succeeds 

  File "cfref.py", line 70, in test_msv_capture
    msv.reportMessage('TEST',7,'This is a test message') 
TypeError: none of the 5 overloaded methods succeeded. Full details:
  void IMessageSvc::reportMessage(const Message& msg, int outputLevel) =>
    takes at most 2 arguments (3 given)
  void IMessageSvc::reportMessage(const Message& message) =>
    takes at most 1 arguments (3 given)
  void IMessageSvc::reportMessage(const StatusCode& code, const string& source = "") =>
    takes at most 2 arguments (3 given)
  (file "", line 0) basic_ios::clear (C++ exception)
  (file "", line 0) basic_ios::clear (C++ exception)

    """
    import GaudiPython
    app = GaudiPython.AppMgr(outputlevel=5)
    msv = app.service('MessageSvc', GaudiPython.gbl.IMessageSvc)

    cm = CaptureMsv()
    assert msv.__class__.__name__ == 'IMessageSvc'
    msv.reportMessage('TEST',7,'This is a test message') 
    cm()
    print cm

    cf = CfRef( __file__ + ".msv_capture.ref" )
    cf(cm)
    if len(cf) == 0:print "matched"
    else:print cf



def test_msv_base():
    collect = Collect()

    import GaudiPython
    app = GaudiPython.AppMgr(outputlevel=5)
    csb = GaudiPython.CallbackStreamBuf(collect)
    ost = GaudiPython.gbl.basic_ostream('char','char_traits<char>')(csb)
    msv = app.service('MessageSvc', GaudiPython.gbl.IMessageSvc)
    assert msv.__class__.__name__ == 'IMessageSvc'

    ori = msv.defaultStream()
    msv.setDefaultStream(ost)

    msv.reportMessage('TEST',7,'This is a test message') 
    msv.reportMessage('TEST',7,'This is a test message') 
    msv.reportMessage('TEST',7,'This is a test message') 

    msv.setDefaultStream(ori) 

    print "collected [%s] " % collect


    cf = CfRef( __file__ + ".msv_base.ref" )
    cf( collect.as_list() )
    if len(cf) == 0:print "matched"
    else:print cf



class CaptureStdout(list):
    def __init__(self, collect=None ):
        if not(collect):
            collect = Collect()
        self.collect = collect
        sys.stdout = self.collect
    def __call__(self):
        sys.stdout = sys.__stdout__ 
        for l in self.collect.as_list(): self.append(l)
    def __repr__(self):
        return "".join(["%-4d %s" % (n,l)  for n,l in enumerate(self)] )
         

class CfRef(list):
    def __init__(self, path ):
        self.path = path
    def __call__(self, s ):
        if os.path.exists(self.path):
            print "comparing ... %s " % self.path 
            self.compare_(s)
        else:
            print "saving ... %s " % self.path
            self.save_(s)
    def save_(self, s ):
        ref = file(self.path,"w")
        ref.writelines( s )
        ref.close()
    def compare_(self, s ):
        from difflib import unified_diff
        for l in unified_diff( file(self.path,"r").readlines() , s ):
            self.append( l )
    def __repr__(self):
        return "".join(self)


def msg(arg, date=False):
    from datetime import datetime
    if date:print datetime.now()
    print arg

def test_cfref_stdout():
    cs = CaptureStdout()
    msg("hello")
    msg("hello", date=True)
    cs()
    #print cs
    cf = CfRef( __file__ + ".stdout.ref" )
    cf(cs)
    if len(cf) == 0:print "matched"
    else:print cf


if __name__=='__main__':
    test_msv_base()
    #test_msv_capture()
    test_cfref_stdout()

