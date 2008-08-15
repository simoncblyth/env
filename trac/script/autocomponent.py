"""
    Defining a set of components using inputs of preexisting components
    with particular name features and the svn properties on directories
    in the repository.
    
    Components with names that end "//" are taken as bases to launch recursive
    walks from, nodes named "tags" or "branches" are not followed, otherwise
    all folder type child nodes are traversed.  Each folder is checked for an
    OWNER property, and the paths of all folders with owners are recorded in 
    a list. 

    Then need to keep 2 trees in sync ...
       distinguish components that are under auto management from 
       hand crafted components by a double slash "//" in the name of the auto ones

     usage ... ON NON PRODUCTION NODE WITH RECOVERED ENVIRONMENT ONLY      
       sudo python $ENV_HOME/trac/script/autocomponent.py $SCM_FOLD/tracs/dybsvn blyth

"""

import sys
import os

from trac.admin.console import TracAdmin
from trac.ticket.model import *

import trac.env
from trac.versioncontrol.api import NoSuchNode, NoSuchChangeset

AUTOMARK = '//'   ## marker that distinguishes an automanaged component
OWNER_PROPNAME = 'owner'
DEFAULT_OWNER = 'offline'

def path2compname(path):
    """  
       Map a repository path to a component name
           dybgaudi/trunk/Simulation/GenTools 
           dybgaudi//Simulation/GenTools
    """
    elem = [ e for e in path.split('/') if e != 'trunk' ]
    if len(elem)>1:
        return elem[0] + AUTOMARK + '/'.join(elem[1:])
    else:
        return elem[0] + AUTOMARK


def compname2path(name):
    """
          Map a component name to a repository path
             dybgaudi//Simulation/GenTools
             dybgaudi/trunk/Simulation/GenTools
   
           This attempts to be independent of what characters are used in the AUTOMARK
        
            BUT theres a whacking great assumption concerning repository path layout 
           
    """    
    pair = name.split(AUTOMARK)
    if len(pair)==2:
        elem = [ e for e in  pair[0].split('/') + pair[1].split('/') if e != '']
        return '/'.join([ elem[0] ,'trunk' ] + elem[1:])  
    else:
        return name



class Comp:

    @classmethod
    def cf(cls, a, b ):
        if a == b:
            return "Comp.cf match %s %s " % ( a , b )
        else:
            return "Comp.cf MISMATCH %s %s " % ( a, b )

    @classmethod
    def from_node(cls, node):
        """ from a repository node to a tuple with the component name and owner """
        if node==None:
            return None
        props = node.get_properties()
        owner = props.get(OWNER_PROPNAME, None)
        name = path2compname(node.path)
        #print "Comp.from_node node.path %s name %s owner %s " % ( node.path, name, owner )         
        return Comp( name, owner, node )

    def __init__(self, name, owner, node=None):
        self.name = name
        self.owner = owner
        self.node = node
        
    def __eq__(self, other):
        return other != None and self.name == other.name and self.owner == other.owner
    def __ne__(self,other):
        return not( self.__eq__(other) )
        
    def path(self):
        return compname2path(self.name)
          
    def pathcheck(self):
        if self.node!=None:
            if self.node.path == self.path():
                return "ok"
            else:
                return " **** PATHCHECK FAILS node.path %s path() %s " % ( self.node.path , self.path() )  
        else:
            return ""

        
    def __repr__(self):
        return "<Comp %s:%s    %s:%s >" % ( self.name,  self.owner ,  self.pathcheck(), self.path() )



class Env:
    """  component list """
    def __init__(self, envdir ):
        admin = TracAdmin()
        admin.env_set(envdir) 
        self.admin = admin
        self.comps = []
        self._get_components()
        
    def _get_components( self ):
        for c in Component.select(self.admin.env_open()):
            if c.name.find(AUTOMARK)>-1:
                self.comps.append( Comp(c.name,c.owner) )

    def print_comps(self):
        for c in self.comps:
            print c

    def find_comp(self, name):
        """ returns the first component with matching name or None if not found """
        for c in self.comps:
            if c.name == name:
                return c
        return None

    def add_comp(self, comp ):
        """ 
              adding a component at depth also adds the intervening containing components 
              ... if not already existing, with owner set to a default value 
            
        """
        pass


    def __repr__(self):
        return "<Env ncomp %s>" % len(self.comps)



class entry(object):
    """ shadow of a node in the repository """
    __slots__ = 'name rev kind isdir path content_length'.split()
    def __init__(self, node):
        for f in entry.__slots__:
            setattr(self, f, getattr(node, f))
    def __repr__(self):
        return "<entry name %s path %s rev %s kind %s isdir %s   >" % ( self.name, self.path , self.rev, self.kind , self.isdir  )


class Repository:
    """   repository tree """
    def __init__(self, envdir, authname):
        trac_env = trac.env.open_environment(envdir)
        repos = trac_env.get_repository(authname)
        self.repos = repos
        self.select = []

    def current_node(self, path ):
        repos = self.repos
        rev = None   ## means the latest    
        node = None
        try:
            if rev:
                rev = repos.normalize_rev(rev)
            rev_or_latest = rev or repos.youngest_rev
            node =  repos.get_node(path, rev_or_latest ) 
        except NoSuchChangeset, e:
            raise ResourceNotFound(e.message, _('Invalid Changeset Number'))
        return node
    
    def walk(self, path='/'):
        print "Repository.walk path %s " % path
        node = self.current_node(path)
        self._walk(node)
    
    def _walk(self, node):
        """ recursive tree walker ... sticking to directories and not following branches or tags 
            and making a selection of directory nodes that have an OWNER property 
        """
        if not(node):
            return
        if node.name in ['branches','tags']:
            return
        if node.isdir:
            props = node.get_properties()
            if props.has_key(OWNER_PROPNAME):
                self.select.append(node)
        if node.isdir:
            for n in node.get_entries():
                self._walk(n)

    def print_selected(self):
        for node in self.select:
            props = node.get_properties()
            enode = entry(node)
            print "%s %s " % ( enode , props[OWNER_PROPNAME] )









def autocomp(args):
    """ 
        seed components have names that end with the AUTOMARK 
        and are walked picking up all owned folder nodes
        ... in this way the state of the repository is gleaned
         
           
    """
    
    envdir = args[0]
    authname = args[1]
    
    
    print "===> initial auto component list (containing %s )  " % AUTOMARK 
    env = Env(envdir)
    env.print_comps()

    print "===> use seed components (ending in %s) to direct the repository walk, selecting owned nodes " % AUTOMARK
    repos = Repository(envdir, authname)
    for c in env.comps:
        if c.name.endswith(AUTOMARK):
            repos.walk( c.path() )
    repos.print_selected()
        
    print "===> checking if %s owned folders are represented in the auto components " % len(repos.select) 
    for node in repos.select:
        #print node   ## trac.versioncontrol.svn_fs.SubversionNode
        cn = Comp.from_node( node )
        
        name = path2compname( node.path )
        comp = env.find_comp( name )
        if comp==None:
            print " comp that needs adding ... %s  " % ( cn )
        else:
            print " cn vs comp %s " % Comp.cf(cn,comp)
            
    
    print "===> checking %s auto components to see if corresponding repository paths still exist " % len(env.comps)
    for c in env.comps:
        node = repos.current_node( c.path() )
        cn = Comp.from_node( node )
        if node==None:
            print "orphaned component %s path not in repository " % ( cn )
        else:
            print " cn vs comp %s " % Comp.cf(cn,c)
    




if __name__=='__main__':
    autocomp(sys.argv[1:])







