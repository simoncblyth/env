#!/usr/bin/env python 

import sys

class CfRef(list):
    """
        tryout object wrapping for 
        functionality of cf_stdout.py

        pause/continue not working 

    """
    def __init__(self, path , start=True ):
        self.path = path
        from cStringIO import StringIO
        self.cur = StringIO()
        if start:
            self.start_()

    def captured(self):return ["%s\n" % l  for l in self.cur.getvalue().split("\n")]
    def start_(self):sys.stdout = self.cur    ## start capturing 
    def __call__(self):self.stop_(compare=True)
    def pause_(self):   self.stop_(compare=False)
    def continue_(self):self.start_()

    def stop_(self, compare=True):
        if sys.stdout == self.cur:
            sys.stdout = sys.__stdout__  ## reset stdout 
        if compare:
            self.compare()

    def compare(self):
        import os
        if os.path.exists(self.path):
            self.compare_()
        else:
            self.save_()

    def save_(self):
        ref = file(self.path,"w")
        ref.writelines( self.captured() )
        ref.close()

    def compare_(self):
        from difflib import unified_diff
        for l in unified_diff( file(self.path,"r").readlines() , self.captured() ):
            self.append( l )
         


def msg(arg, date=False):
    from datetime import datetime
    if date:
        print datetime.now()
    print arg


if __name__=='__main__':

    cf = CfRef( __file__ + ".ref" )

    msg("hello")
    msg("hello", date=True)

    cf()

    if len(cf) == 0:
        print "matched"
    else:
        for l in cf:
            print l,


