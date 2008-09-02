

import ROOT



class geomap:
    def __init__(self, path=None):
        self.map = None
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
             
    def nod(self,key):
        return self.map.get(key, None)
    
    def vol(self,key):
        n = self.nod( key )
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
        
    
if __name__=='__main__':
    src = "$ENV_HOME/aberdeen/root/WorldWithPMTs.root"
    gm = geomap()
    gm.import_volume( src , "World")
    
    