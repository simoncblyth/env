import os, sys, logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('dybtest.mem')

class Ps(int):
    """
        ps reported RSS converted to bytes or None if cannot determine
        
        From "man ps" :
           * RSS : resident set size, the non-swapped physical memory 
             that a task has used (in kiloBytes).
    
        Inspired by supervisord/superlance memmon listener
    """
    _rsscmd = "ps -orss= -p %s "
    rsscmd = property( lambda self:self._rsscmd % self )
    
    def _rss( self ):
         try:
             r = os.popen( self.rsscmd )
         except OSError, e:
             r = None
             log.warn("oserror from popen %s " % self.rsscmd ) 
         except IOError, e:
             r = None
             log.warn("ioerror from popen %s " % self.rsscmd ) 
         return r and r.read() or None
    rss = property( _rss )

    def _rss_megabytes( self ):
        data = self.rss 
        if not data:
            return None
        try:
            val = data.lstrip().rstrip()
            val = float(val)/(1024)   # rss starts in KB         
        except ValueError:
            val = None
        return val
    rss_megabytes = property( _rss_megabytes )

def rss_megabytes(pid):
    ps = Ps(pid)
    return ps.rss_megabytes

def test_rss():
    pid = os.getpid() 
    rss = rss_megabytes(pid)
    rssmax = 1   
    print rss, pid
    assert rss < rssmax , locals()  

if __name__=='__main__':
    test_rss()
    import sys
    pid = len(sys.argv) > 1 and sys.argv[1] or os.getpid()

    log.info( Ps(pid).rss_megabytes )


