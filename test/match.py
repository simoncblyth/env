import re

checks = {
  '.*FATAL':2,
  '.*ERROR':1,
}
 

class Matcher:
    def __init__(self, patterns , verbose=False ):
        self.patterns = patterns
        self.patns={}
        self.verbose = verbose
        for pt,rc in patterns.items():
            self.patns[pt] = re.compile(pt), rc  
    
    def __call__(self, line):
        """
            provide a status code for the provided line 
        """
        rc = self.match(line)
        if self.verbose: print "[%-1s] %s\n" % ( rc, line.rstrip('\n') ),
        return rc

    def match( self , line ):
        """ return the code of the first match, or zero if no match """
        for pt in self.patns.keys():
             if self.patns[pt][0].match(line)>-1:
                  return self.patns[pt][1]
        return 0     

    def __repr__(self):
        import pprint
        return "<Matcher %s >" % pprint.pformat( self.patterns )
                 

if __name__=='__main__':
    m = Matcher(checks, verbose=True )
    assert m("hello\n") == 0  
    assert m("hello FATAL \n") == 1  
    assert m("hello\n") == 0  
    assert m("hello\n") == 0  
    print m

      

