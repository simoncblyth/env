

import os

class HistRef:
    def __init__(self, path ):
        self.path = path

    def refpath(self):
        dir = os.path.dirname(self.path)
        base = os.path.basename(self.path)
        return os.path.join( dir, "histref_"+base )

    def __call__(self, **kwa ):
        path = self.path
        if not(os.path.exists(path)):
            print "HistRef ABORT no path %s " % path 
            return 1      

        refp = self.refpath()
        if not(os.path.exists(refp)):
            print "HistRef no reference, so blessing path %s as the future reference " % path 
            import shutil
            shutil.copy( "%s %s" % ( path , refp))  
            return 0

        x = XmlRoot(refp)
        y = XmlRoot(path)
        xy = XmlCfRoot( x, y )

    def __repr__(self):
        return "<HistRef %s >" % self.path 



class XmlCfElem:
    def __init__(self, a , b ):
        self.a = a
        self.b = b

    def diff(self):
        from difflib import unified_diff
        df = unified_diff( self.a.lines , self.b.lines )
        return list(df)

    def cf(self):
         nd = len(self.diff())
         if nd>0:
             return "has %s lines of diff " % ( nd )   
         return "matches"

    def __repr__(self):
         return "<XmlCfElem %s > " % ( self.cf() )     



class XmlNamed:

    tnpath = {
      'TH1F':'.//TH1/TNamed'
    }

    def __init__(self, elem , kls ):
        """
             http://effbot.org/zone/element-xpath.htm
        """
        self.name  = self.name_( elem, kls )
        self.lines = self.lines_(elem)

    def name_(self, elem, kls ):
        n,t = self.tnamed( elem, kls )
        return "%s:%s:%s" % ( kls, n, t ) 

    def tnamed( self, elem, kls ): 
        tnpath = XmlNamed.tnpath.get(kls) 
        tn = elem.findall( tnpath )
        name = tn[0].find("./fName").get("str")
        title = tn[0].find("./fTitle").get("str")
        return name, title

    def lines_(self, elem):
         from xml.etree import ElementTree as ET
         from StringIO import StringIO
         s = StringIO()          
         ET.ElementTree(elem).write(s)
         return s.getvalue().split("\n")

    def __repr__(self):
        return "<XmlNamed %s >" % self.name




class XmlRoot:
    """
         Views a root .xml file as a list of objects within the 
         list of klasses
    """ 
    klasses = ['TH1F']

    def __init__(self, path ):
        from xml.etree import ElementTree as ET
        self.root = ET.parse(path).getroot()

    def objects( self):
        return [XmlNamed(o,kls) for kls in XmlRoot.klasses for o in self.root.findall(".//Object") if o.get("class") == kls ]

    def ls( self ):
        for o in self.objects():
            print " %s" % ( o.name )
     


class XmlCfRoot:
    def __init__(self, a, b ):
        self.a = a
        self.b = b
    def cf(self):
        na = len( self.a.objects() )
        nb = len( self.b.objects() )
        assert na == nb
        return [XmlCfElem(s,o) for s,o in zip(self.a.objects(),self.b.objects())]
     
        



def create(path):
    from ROOT import TFile, TH1F
    f = TFile.Open( path ,"recreate")
    for i in range(10):
        h = TH1F("h","test%s" % i ,1000,-2,2)
        h.FillRandom("gaus")
        h.Write()
        h.Delete()
    f.Close()


if __name__=='__main__':
    import os

    x = 'Ex1.xml'
    if not(os.path.exists(x)):
        create(x)
    x = XmlRoot(x)
    x.ls()

    y = 'Ex2.xml'
    if not(os.path.exists(y)):
        create(y)
    y = XmlRoot(y)
    y.ls()


    xy = XmlCfRoot( x, y )
    for cf in xy.cf():
        print cf





