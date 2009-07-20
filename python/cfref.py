#!/usr/bin/env python 


"""

def test11ostreamcallback(self) :
   def test11do_msg(s) : self.got_it = True
   self.got_it = False
   buf  = GaudiPython.CallbackStreamBuf(test11do_msg)
   ostream = GaudiPython.gbl.basic_ostream('char','char_traits<char>')(buf)
   msgSvc = g.service('MessageSvc', GaudiPython.gbl.IMessageSvc)
   original = msgSvc.defaultStream()
   msgSvc.setDefaultStream(ostream)
   msgSvc.reportMessage('TEST',7,'This is a test message')
   msgSvc.setDefaultStream(original)
   self.failUnless(self.got_it)


"""
import sys
import os

class CaptureStdout(list):
    def __init__(self):
        from cStringIO import StringIO
        self.buf = StringIO() 
        sys.stdout =  self.buf
    def __call__(self):
        sys.stdout = sys.__stdout__ 
        for l in self.buf.getvalue().split("\n"): self.append("%s\n" % l)
    def __repr__(self):
        return "".join(["%-4d %s" % (n,l)  for n,l in enumerate(self)] )
         

class CfRef(list):
    def __init__(self, path , captor ):
        self.path = path
        print "%s start capture ... " % self.__class__.__name__ 
        self.capture = captor()

    def __call__(self):
        self.capture()
        print "%s stop capture : collected %s lines " % ( self.__class__.__name__ , len(self.capture) ) 
        if os.path.exists(self.path):
            print "comparing ... %s " % self.path 
            self.compare_()
        else:
            print "saving ... %s " % self.path
            self.save_()

    def save_(self):
        ref = file(self.path,"w")
        ref.writelines( self.capture )
        ref.close()

    def compare_(self):
        from difflib import unified_diff
        for l in unified_diff( file(self.path,"r").readlines() , self.capture ):
            self.append( l )
         


def msg(arg, date=False):
    from datetime import datetime
    if date:
        print datetime.now()
    print arg


if __name__=='__main__':
    cf = CfRef( __file__ + ".ref" , CaptureStdout )

    msg("hello")
    msg("hello", date=True)

    cf()


    print cf.capture


    if len(cf) == 0:
        print "matched"
    else:
        for l in cf:
            print l,


