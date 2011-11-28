
from ROOT import kTRUE, kFALSE

def treelist( tgn , path ):
    p = "%s/%s" % ( path , tgn.GetName() )
    viz = tgn.IsVisible()
    col = tgn.GetVolume().GetLineColor()
    print "%-5s %-5s %s " % ( viz, col , p )
    kids = tgn.GetVolume().GetNodes()
    if kids:
        for child in kids:treelist(child, p)


class TGeoWalk:
    def __init__(self, vm , ed ):
        self.vm = vm
        self.ed = ed
    def __call__(self, tgn ):
        self.walk_(tgn, "")
    def __repr__(self):
        return "\n".join( ["vm"] +  [ str(self.vm) ] ) 
    def walk_(self, tgn , path ):
        n = tgn.GetName()
        p = "%s/%s" % ( path , n )
        tgv = tgn.GetVolume()
        kids = tgv.GetNodes()
        leaf = not(kids) or len(kids) == 0 
        self.vm( n )
        self.vm.update( { 'leaf':leaf , 'path':p } )
        self.ed( self.vm , tgn ) 
        if self.vm.get('pname',None) == 'catchall' :
            print "walk %s " % ( self.vm )
        if kids:
            for child in kids:self.walk_(child, p)

class Matcher(dict):
    def __init__(self):
        self.stats = {}
    def __call__( self, name ):
        self.clear()
        for pname,patn,ovr in self.__class__.patns:
            m = patn.match(name)
            self.incr('count')
            if m:
                self.incr('match')
                self.incr(pname)
                self.update( { 'pname':pname } )
                self.update( m.groupdict() )
                self.update( ovr )
                break
        return self
    def incr(self, label):
        if self.stats.get(label, None):
            self.stats[label] += 1
        else:
            self.stats[label] = 1
    def report(self):
        return  "\n".join([  "%-10s %-7s " % ( k , v ) for k, v in sorted( self.stats.items(), lambda a,b:cmp(a[1],b[1]) ) ])
 

class VolEditor:
   def __call__(self, d , tgn ):
       #print "ved called for %s %s leaf:%s " % ( tgv.GetName() , d , leaf )
       self.color( d, tgn ) 
       self.vizib( d, tgn ) 

   def color(self, d, tgn ):
       tgv = tgn.GetVolume() 
       tco = d.get('color', None )
       if tco:
           print "setting color of %s to %s " % ( tgv , tco )
           tgv.SetLineColor(col)
       else:
           print "no color for %s " % d 

   def vizib(self, d, tgn ):
       """
          default is to set leafs visible and containers invisible
          .. overridable with a viz attribute   
       """
       leaf = d.get('leaf', None )
       if leaf == None:
          print "leaf is none " 
       elif leaf == True :
          print "setting visible for %s " % tgn.GetName() 
          tgn.SetVisibility(kTRUE)  
       elif leaf == False :
          print "setting in-visible for %s " % tgn.GetName() 
          tgn.SetVisibility(kFALSE)  

       viz = d.get( 'viz' ,  None )
       if viz:
           print "viz override for %s " % tgv.GetName()
           tgn.SetVisibility(viz)
       

