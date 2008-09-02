
import ROOT
import re

class geomap:
    def __init__(self, path=None):
        self.map = {}
        self.selection = []
        if path != None:
            self.import_(path)
    
    def import_(self, path):
        ROOT.TGeoManager.Import( path )
        self.create_map()
        
    def import_volume(self, path, vname):
        top = ROOT.TGeoVolume.Import( path , vname )
        ROOT.gGeoManager.SetTopVolume( top )
        ROOT.gGeoManager.CloseGeometry()
        self.create_map()
        
    def create_map(self):
        tn = ROOT.gGeoManager.GetTopNode()
        self.walk(tn, "")

    def walk(self, node, path ):
        if node!=None:
            name = node.GetName()
        else:
            name = "NULL"
        key = self.unique_key( name )
        
        self.map[key] = node
        vol = node.GetVolume()
        nodes = vol.GetNodes()
        
        if nodes != None:
            for n in nodes:
                self.walk( n , "%s/%s" % ( path , name ) )

          
    def unique_key(self, key ):
        assert len(key)<100, "key length too long ... recursing out of control ? "
        if not(key in self.map):
            return key
        else:
            return self.unique_key( "%sx" % key )
             
           
            
    def node(self,key):
        return self.map.get(key, None)
    
    def volume(self,key):
        n = self.node( key )
        if n != None: return n.GetVolume()
        return None
       
    def select(self, patn=None ):
        """ control the selection by the regexp pattern """
        if patn==None:
            self.selection = self.map.keys()
        else: 
            self.selection = []
            p = re.compile( patn )
            for k in self.map.keys():
                if p.match(k):
                    self.selection.append(k) 
	print "selection with patn %s matches %s keys out of possible %s " % ( patn , len(self.selection), len(self.map) )
        return self.selection        

    def apply_to_v(self, func=lambda v:v ):
        print "apply function %s to %s selected volumes " % ( func , len(self.selection) )
	for k in self.selection:
            v = self.volume(k)
            if v != None:
                func(v)
        
    def set_line_color(self, patn,  col ):
        self.select(patn)
        self.apply_to_v( lambda v:v.SetLineColor(col) )
       
    def set_visibility(self, patn , viz ): 
        self.select(patn)
        self.apply_to_v( lambda v:v.SetVisibility( viz ) )
    



if __name__=='__main__':
    src = "$ENV_HOME/aberdeen/root/WorldWithPMTs.root"
    gm = geomap()
    gm.import_volume( src , "World")

    #print gm.map
    gm.set_visibility( None, False )
    gm.set_visibility( ".*Tube" , True )      

    tn = gm.node("World_1")

    ROOT.TEveManager.Create() 
    etn = ROOT.TEveGeoTopNode(ROOT.gGeoManager, tn )
    ROOT.gEve.AddGlobalElement(etn)
    ROOT.gEve.Redraw3D(True)

    
    
    
