"""
  Parse xml reports output by nosetests --with-xml-output 
  into an informative text message on stdout, 
  for emailing in fail mails

  Usage :
     python report.py path/to/test-report.xml
     python report.py /tmp/slave/i686-slc46-gcc346/build_dybinst_965/test-rootio.xml

"""

class Test:
    def __init__(self, t ):
        self.t = t 

    def hdr( self ):
        hdr = '#' * 10 
        for k in 'name status duration file'.split():
            hdr += " %s:%s " % ( k , self.t.get(k) ) 
        hdr += '#' * 10 
        return hdr

    def isfail(self):
        return self.t.get('status') == 'failure'    

    def summary(self):
        if self.isfail():
            print "%s\n" % self.hdr()
            print self.t.findtext('stdout')


def status_report(path):
    import xml.etree.cElementTree as et
    try:
        r = et.parse(path).getroot() 
    except:
        print "ERROR: failed to parse %s " % path
        return 3 

    for t in r:
        test = Test(t) 
        test.summary()
    return 0


def main(args):
    if len(args)==0:
        return 1
    
    import os
    path = args[0]
    if not(os.path.exists(path)):
        print "ERROR: no such path %s " % path
        return 2 
    return status_report( path )   
   

if __name__=='__main__':
    import sys
    rc = main(sys.argv[1:])
    if rc!=0:
        print sys.modules[__name__].__doc__
    sys.exit(rc) 




