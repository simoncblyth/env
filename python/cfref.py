#!/usr/bin/env python 

import sys

class CfRef(list):
    """
        tryout object wrapping for 
        functionality of cf_stdout.py

    """
    def __init__(self, path , start=True ):
        self.path = path
        if start:
            self.start_()

    def start_(self):
        from cStringIO import StringIO
        self.cur = StringIO()
        sys.stdout = self.cur    ## start capturing 

    def __call__(self):self.stop_()

    def stop_(self, compare=True):
        if sys.stdout == self.cur:
            sys.stdout = sys.__stdout__  ## reset stdout 
        if compare:
            self.compare()

    def captured(self):
        return ["%s\n" % l  for l in self.cur.getvalue().split("\n")]

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
        refl = file(self.path,"r").readlines()
        from difflib import unified_diff
        for l in unified_diff( refl , self.captured() ):
            self.append( l )
         





if __name__=='__main__':

    cf = CfRef( __file__ + ".ref" )

    print "hello world "
    from datetime import datetime
    print datetime.now()

    cf()

    if len(cf) == 0:
        print "matched"
    else:
        for l in cf:
            print l,


