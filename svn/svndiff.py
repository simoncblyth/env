#!/usr/bin/env python
"""
Compare "svn diff" with "svnlook diff" on C2 ... 

  sudo cp -rf  /var/scm/repos/env /tmp/
  sudo chown -R blyth.blyth /tmp/env

  svnlook diff /tmp/env > /tmp/svnlook   
  ( cd ~/env ; svn diff -c $(svnversion))  > /tmp/svndiff
  diff /tmp/svnlook /tmp/svndiff

       * differences in the "chrome" only

"""
import os, re

class Text(list):
    """
    Base class that holds lines of text as a list and
    provides facility for splitting the text and verifying 
    can recreate the unsplit from the pieces 
    """
    def __init__(self, *args, **kwargs):
        list.__init__(self, args[0] )
        self.meta = {}
        self.meta.update(kwargs)
        self.divs = []
        self.children = []

    def split_(self, cls, predicate=lambda line:True, offset=0 ):
        divs = []
        for n,line in enumerate(self):
            if predicate(line):
                divs.append(n+offset)
        self.divs = divs
        children = []
        for i, n in enumerate(divs):
            if i + 1 < len(divs):
                m = divs[i+1]
            else:
                m = len(self) 
            child = cls(self[n:m], index=i, begin=n, end=m)
            children.append(child) 
        return children

    def _smry(self):
        if len(self.children)>0:
            return "\n" + "\n".join([repr(c) for c in self.children] )
        else:
            return ""
    smry = property(_smry)

    def __str__(self):
        return "\n".join(self)
    def __repr__(self):
        return "%s %r %r" % ( self.__class__.__name__, self.meta, self.divs ) + self.smry

    rejoin = property(lambda self:"\n".join([str(c) for c in self.children]))

    def check(self, verbose=False, hdr=None ):
        """Check can put together the split text """ 
        rejo = self.rejoin 
        if hdr:
            rejo = "\n".join([ hdr, rejo])
        if verbose:
            print "." * 100
            print str(self)
            print "." * 100
            print rejo
            print "." * 100
        assert str(self) == rejo, ("mismatch for %s %r " % ( self.__class__.__name__, self), rejo , str(self) ) 


class Hunk(Text):
    """
    I define a hunk to be a stretch of diff text that shares the same first character...
    """
    def __init__(self, *args, **kwargs):
        Text.__init__(self, *args, **kwargs)



class Block(Text):
    ptn = re.compile("^@@ (?P<ablk>[-\+,\d]*) (?P<bblk>[-\+,\d]*) @@$")
    def __init__(self, *args, **kwargs):
        Text.__init__(self, *args, **kwargs)
        self.children = []
        self.parse_hdr()
        self.parse_body()
        self.check(verbose=False)

    hdr = property(lambda self:self[0])
    def parse_hdr(self):
        m = self.ptn.match(self[0])   
        assert m, ( "failed to match %s " % self[0] )
        self.meta.update( m.groupdict() )

    def parse_body(self, verbose=False):
        """
        Look for contiguous Hunks of text with the same first character 
        """
        l,start  = "<",0
        for i,line in enumerate(self + [">"]):
            if len(line)>0:
                c = line[0]
            else:
                c = "."
            assert c in " >@+-."
            if verbose:
                print "[%2d, %s,%s,%d] %s " % ( i,c,l,start, line)
            if c == l:
                pass
            else:
                ## record prior when transition to new contiguos first char ... but avoid fake initial hunk
                if l == "<":
                    pass
                else:
                    self.children.append( Hunk( self[start:i], c=l ,begin=start,end=i ))
                l = c
                start = i
        #for hnk in self.children:
        #    print repr(hnk)   
        #    print hnk   



class Delta(Text):
   """
   Hold raw text of a single difference ... 
   split into sub-Blocks using the block divider
   """
   req = 'label path0 div apath arev bpath brev'.split()
   ptn = (
           re.compile("^(?P<label>\S*): (?P<path0>\S*)"),
           re.compile("^(?P<div>===================================================================)"),
           re.compile("^--- (?P<apath>\S*)\t\(rev\S* (?P<arev>\d*)\)"),
           re.compile("^\+\+\+ (?P<bpath>\S*)\t\(rev\S* (?P<brev>\d*)\)"),
         )

   hdr = property(lambda self:"\n".join(self[0:4]))

   def __init__(self, *args, **kwargs):
       Text.__init__(self, *args, **kwargs)
       self.parse_hdr()
       self.children = self.split_(Block, lambda l:Block.ptn.match(l), offset=0 )  ## offset controls where to divide ...  
       self.check(hdr=self.hdr)

   def parse_hdr(self):
       """Line by line pattern matching of the header """
       for i,ptn in enumerate(self.ptn):
           line = self[i]
           m = self.ptn[i].match( line )
           assert m, ( "failed to match %s " % line )
           self.meta.update( m.groupdict() )
       for req in self.req:
           assert req in self.meta, "required match parameter not found %s " % req
       pass
       del self.meta['div']


class Diff(Text):
    """
    Hold the raw text of the full output of "svn diff" or "svnlook diff"
    and split into sub-Delta using the divider
    """
    def __init__(self, *args, **kwargs ):
        Text.__init__(self, *args, **kwargs)
        self.children = self.split_(Delta,lambda l:l == "===================================================================", offset=-1 )
        self.check()

    def dump(self):
        for dlt in diff.children:
            print repr(dlt)
            for blk in dlt.children:
                print repr(blk)
                print blk
                for hnk in blk.children:
                    print repr(hnk)
                    print hnk
        

class SVNDiff(dict):
    _cmd = "svn diff -c %(rev)s"
    cmd = property(lambda self:self._cmd % self)
    def __call__(self, **kwargs):
        self.update(kwargs)
        return os.popen(self.cmd).read()



if __name__=='__main__':

   sd = SVNDiff()

   for rev in (3010,3009,3008,3007,):
       txt = sd(rev=rev)
       diff = Diff(txt.split("\n"))
       print repr(diff)        

          
