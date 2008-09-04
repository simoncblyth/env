import re

class Matcher:
    def __init__(self, patterns , verbose=False ):
        self.patns={}
        self.verbose = verbose
        for pt,rc in patterns.items():
            self.patns[pt] = re.compile(pt), rc  
    
    def __call__(self, line):
        """
            provide a status code for the provided line 
        """
        rc = self.match(line)
        if self.verbose: print "[%-1s] %s" % ( rc, line ),
        return rc

    def match( self , line ):
        """ return the code of the first match, or zero if no match """
        for pt in self.patns.keys():
             if self.patns[pt][0].match(line)>-1:
                  return self.patns[pt][1]
        return 0     
                 

if __name__=='__main__':
    checks = { '.*FATAL':1 }
    m = Matcher(checks, verbose=True )
    assert m("hello\n") == 0  
    assert m("hello FATAL \n") == 1  
    assert m("hello\n") == 0  
    assert m("hello\n") == 0  
    print m

      

