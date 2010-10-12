"""
  Parse xml reports output by nosetests --with-xml-output 
  into an informative text message on stdout, 
  for emailing in fail mails

  Usage :
     python report.py path/to/test-report.xml
     python report.py /tmp/slave/i686-slc46-gcc346/build_dybinst_965/test-rootio.xml

"""

class Test(object):
    def __init__(self, t ):
        self.t = t 

    def _hdr( self ):
        hdr = '#' * 10 
        hdr += "".join( [" %s:%s " % ( k , self.t.getAttribute(k) ) for  k in 'name status duration file'.split()])
        hdr += '#' * 10 
        return hdr
    hdr = property( _hdr )

    def _stdout(self):
        return self.t.getElementsByTagName("stdout")[0].childNodes[0].nodeValue     
    stdout = property( _stdout )
    isfail = property( lambda self:self.t.getAttribute('status') == 'failure' )   
    def __repr__(self):
        return "\n".join( [self.hdr, self.isfail and self.stdout or "" ]) 


def status_report(path):
    from xml.dom.minidom import parse
    try:
        tests = parse(path).getElementsByTagName("test") 
    except:
        return "ERROR: failed to parse %s " % path
    return "\n".join([repr(t) for t in filter( lambda _:_.isfail, map(lambda _:Test(_),tests))])


def main(args):
    assert len(args)==1
    print status_report( args[0] )   
   
if __name__=='__main__':
    import sys
    sys.exit(main(sys.argv[1:]))




