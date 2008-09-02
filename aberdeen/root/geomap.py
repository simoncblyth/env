

import ROOT



class geomap:
    def __init__(self, path=None):
        self.map = {}
        self.sel = []
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
        
        
    def select(self, patn=None ):
        """ control the selection by the regexp pattern """
        if patn==None:
            self.sel = sel.map.keys()
            return
            
        p = re.compile( patn )
        for k in self.map.keys():
            if p.match(k):
                self.sel.append(k) 
        
    def apply_to_v(self, func=lambda v:v ):
        for k in self.sel:
            v = self.vol(k)
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
    
    tn = gm.nod("World_1")
    etn = ROOT.TEveGeoTopNode(ROOT.gGeoManager, tn )

    ROOT.TEveManager.Create()  ## put up the GUI
    from ROOT import gEve      ## this only works after creation 
    
    gEve.AddGlobalElement(etn);
    gEve.Redraw3D(True)

    gm.set_visibility( None, False )
    gm.set_visibility( "Tube" , True )      
    

    
    
    