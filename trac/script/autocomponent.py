"""
    Defining a set of components using inputs of preexisting components
    with particular name features and the svn properties on directories
    in the repository.
    
    Components with names that end "//" are taken as bases to launch recursive
    walks from, nodes named "tags" or "branches" are not followed, otherwise
    all folder type child nodes are traversed.  Each folder is checked for an
    OWNER property, and the paths of all folders with owners are recorded in 
    a list. 


    Ownwer properties are read ony ...



    Then need to keep 2 trees in sync ...
       distinguish components that are under auto management from 
       hand crafted components by a double slash "//" in the name of the auto ones

     usage ... ON NON PRODUCTION NODE WITH RECOVERED ENVIRONMENT ONLY      
       sudo python $ENV_HOME/trac/script/autocomponent.py $SCM_FOLD/tracs/dybsvn blyth




   component manipulation is fast ... so avoid the duplication of maintaining 
    the component state in Env, go direct to the component model ????



"""

import sys
import os

import trac.env

from trac.admin.console import TracAdmin
from trac.ticket.model import Component
from trac.resource import ResourceNotFound
from trac.versioncontrol.api import NoSuchNode, NoSuchChangeset

from scheme import SubTrunkAutoComponent, BaseTrunkAutoComponent



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
    def __init__(self, envdir, authname, schema):
        trac_env = trac.env.open_environment(envdir)
        repos = trac_env.get_repository(authname)
        self.schema = schema
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
        """ selects directory nodes that have owner property set and their parent directories """
        print "Repository.walk path %s " % path
        node = self.current_node(path)
        self._walk(node)
        
    def is_selected(self,path):
        for node in self.select:
            if node.path == path:
                return True
        return False
    
    def parent_select(self, path):
        """  expand selection to cover the parent of a node , if not already selected """
        elem = path.split('/')
        if len(elem)==0:
            return
        for i in range(len(elem)-1,0,-1):
            pp = '/'.join(elem[0:i])
            if not(self.is_selected(pp)):
                pnode = self.current_node(pp) 
                if pnode.isdir:
                    self.select.append(pnode)
    
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
            if self.schema.prop_select( props ):
                self.select.append(node)
                self.parent_select(node.path)
        
        if node.isdir:
            for n in node.get_entries():
                self._walk(n)

    def print_selected(self):
        for node in self.select:
            props = node.get_properties()
            enode = entry(node)
            print enode





def Component__repr__(self):
    return "<Component %s:%s >" % (self.name, self.owner )

Component.__repr__ = Component__repr__


def autocomp(args):
    """ 
        seed components have names that end with the AUTOMARK 
        and are walked picking up all owned folder nodes
        ... in this way the state of the repository is gleaned
         
    """
    envdir = args[0]
    authname = args[1]
    
      
    admin = TracAdmin()
    admin.env_set(envdir) 


    if envdir.endswith("dybsvn"):
        acs = SubTrunkAutoComponent(  "owner", "offline" , "//" )
    else:
        acs = BaseTrunkAutoComponent( "owner" , "blyth" , "//" )

    print acs


    print "===> use seed components to direct the repository walk, selecting owned nodes .. and their parents "
    repos = Repository(envdir, authname, acs)
    for c in Component.select(admin.env_open()):
        if acs.is_seedcomp(c.name):
            print "walk seed comp %s " % c
            repos.walk( acs.compname2path(c.name) )
        else:
            print "skip non seed comp %s " % c
                    
    repos.print_selected()
        
    print "===> elevating %s owned folders to components or updating owners " % len(repos.select) 
    for node in repos.select:
        #print node   ## trac.versioncontrol.svn_fs.SubversionNode
        props = node.get_properties()
        owner = props.get( acs.name, acs.default )
        name = acs.path2compname(node.path)
        
        try:
            c = Component( admin.env_open(), name)
            if c.owner == owner:
                print "owner remains same : %s " % c 
            else:
                print "owner changed to %s : formerly %s " % ( owner , c ) 
                c.owner = owner
                c.update()
        except ResourceNotFound:
            c = Component( admin.env_open() )
            c.name = name
            c.owner = owner
            print "inserting component %s  " % ( c )
            c.insert()        
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
              

    print "===> purge auto components without corresponding repository paths  " 
    for c in Component.select(admin.env_open()):
        if acs.is_autocomp(c.name):
            path = acs.compname2path( c.name )
            node = repos.current_node( path )
            if node==None:
                print "orphaned component %s  path %s not in repository ... delete " % ( c , path )
                c.delete()
            else:
                print "component %s is valid " % c 
    



if __name__=='__main__':
    autocomp(sys.argv[1:])







