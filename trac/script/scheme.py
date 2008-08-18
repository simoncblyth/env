


class AutoComponent(object):
    """
        stub interface for path <--> component name mapping and detection 
        of managed and seed components 
        
        a seed component is one that gets recursively walked to select directory 
        nodes of interest
        
        a managed component is one that can be created/deleted/edited by 
        the autocomponent scrpt
        
    """

    def __init__(self, name, default, mark):
        self.name = name
        self.default = default
        self.mark = mark
        
    def path2compname(self,path):raise NotImplementedError
    def compname2path(self,name):raise NotImplementedError
    def is_autocomp(self,name):raise NotImplementedError
    def is_seedcomp(self,name):raise NotImplementedError
        
    def prop_select(self, props):
        return props.has_key(self.name)
    def prop_get(self, props):
        return props.get( self.name, self.default )
    def __repr__(self):
        return "<%s %s %s %s>" % (self.__class__.__name__, self.name , self.default , self.mark )




class SubTrunkAutoComponent(AutoComponent):

    def __init__(self, name, default, mark):
        AutoComponent.__init__(self, name, default, mark)
        
    def path2compname(self, path):
        """  
            Map a repository path to a component name
              dybgaudi/trunk/Simulation/GenTools 
              dybgaudi//Simulation/GenTools
        """
        elem = [ e for e in path.split('/') if e != 'trunk' ]
        if len(elem)>1:
            return elem[0] + self.mark + '/'.join(elem[1:])
        else:
            return elem[0] + self.mark

    def compname2path(self, name):
        """
          Map a component name to a repository path
             dybgaudi//Simulation/GenTools
             dybgaudi/trunk/Simulation/GenTools
   
           This attempts to be independent of what characters are used in the AUTOMARK
        
            BUT theres a whacking great assumption concerning repository path layout 
           
        """    
        pair = name.split(self.mark)
        if len(pair)==2:
            elem = [ e for e in  pair[0].split('/') + pair[1].split('/') if e != '']
            return '/'.join([ elem[0] ,'trunk' ] + elem[1:])  
        else:
            return name

    def is_autocomp(self, name):
        return name.find(self.mark)>-1

    def is_seedcomp(self, name):
        return name.endswith(self.mark)
        


class BaseTrunkAutoComponent(AutoComponent):
    def __init__(self, name, default, mark):
        AutoComponent.__init__(self, name, default, mark) 
        
    def path2compname(self, path):
        """  
                 trunk/trac/script
             -->  //trac/script
        """
        elem = [ e for e in path.split('/') if e != 'trunk' ]
        name = self.mark + '/'.join(elem)
        return name
    
    def name2elem(self,name):
        n = name[len(self.mark):]
        return n.split('/')
        
    def compname2path(self, name):
        assert self.is_autocomp(name),  " for name [%s][%s] " % ( name , self.mark )
        elem = self.name2elem(name)
        path =  '/'.join( [ 'trunk' + elem ] )
        return path
        
    def is_autocomp(self, name):
        return name.find(self.mark)==0
        
    def is_seedcomp(self, name):
        elem = self.name2elem(name)
        return self.is_autocomp(name) and len(elem) == 1
         






