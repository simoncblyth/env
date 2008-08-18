"""
   
    This script allows semi-automated management of a subset of the 
    Trac ticket components by means of setting svn properties on the directories
    in the repository.  The directories to walk in the repository are 
    controlled by special characteristics of the component names and similarly 
    components are identified as being under management (and thus susceptable to 
    edits/deletion) by characteristics of the name.
    
    These specific characteristics and the mappings between repository paths and
    component names are specified the the AutoComponent scheme class. With the scheme in 
    force depending on the basename of the environment directory passed.
      
    Usage :
        1) manually create seed "components" in the web interface 
               http://localhost/tracs/env/admin/ticket/components
           with names based on repository paths that will be selected as seeds, 
           these directories will be walked looking for folders with 
           owner properties set. Such directories and their parent directories are selected.
           
            eg creating components :
                   "//dybpy" ("BaseTrunk" scheme used in env repo)
                "dybgaudi//"  ("SubTrunk" scheme used in dybsvn repo)
           
        2) run the script on the target trac environment directory 
           
              sudo python $ENV_HOME/trac/autocomp/autocomponent.py $SCM_FOLD/tracs/env blyth
              sudo python $ENV_HOME/trac/autocomp/autocomponent.py $SCM_FOLD/tracs/dybsvn blyth
              sudo python $ENV_HOME/trac/autocomp/autocomponent.py $SCM_FOLD/tracs/dybsvn ntusync
             OR more simply 
               trac-; autocomp-; autocomp-sync <name-defaulting-to-TRAC_INSTANCE>  
           
            sudo is needed as the trac log file is written to 
    
           The tree of paths selected is used to update the list of trac ticket components
           with additions of new components, deletion of components which no longer have associated
           repo paths, and ownership changes.
            
               
        3) examine the component list in the web interface   
                http://localhost/tracs/env/admin/ticket/components
           or in the pull down menu on creating/editing a ticket
   
    
    NB 
       a)  Owner properties are treated read ony , they are never written by this script
       b)  only components with names fulfiling the managed component criteria (ofter containing "//" )
           are touched by this script 



    WHILE TESTING ... USE ON NON PRODUCTION NODE WITH RECOVERED ENVIRONMENT ONLY      
          sudo python $ENV_HOME/trac/autocomp/autocomponent.py $SCM_FOLD/tracs/dybsvn blyth


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
        """  
           expand selection to cover the parents of a node , if not already selected ... selecting 
           back up to the seed node
        """
        elem = path.split('/')
        if len(elem)==0:
            return
        for i in range(len(elem)-1,0,-1):
            pp = '/'.join(elem[0:i])
            if not(self.is_selected(pp)):
                pnode = self.current_node(pp) 
                if pnode.isdir:
                    self.select.append(pnode)
                
                name = self.schema.path2compname(pp)
                if self.schema.is_seedcomp(name):
                    print "%s parent_select [%s] [%s] stop selection above this seed node " % ( self, pp , name ) 
                    return
    
    
    def _walk(self, node):
        """ recursive tree walker ... sticking to directories and not following branches or tags 
            and making a selection of directory nodes that have an OWNER property 
        """
        if not(node):
            return
            
        if node.name in ['branches','tags']:
            return
        if node.isdir:
            print "_walk %s " % node.path
            props = node.get_properties()
            if self.schema.prop_select( props ):
                self.select.append( node )
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
    seed_comps = [ c.name for c in Component.select(admin.env_open()) if acs.is_seedcomp(c.name) ]
    print "seed_comps: %s " % seed_comps 
    
    repos = Repository(envdir, authname, acs)    
    for name in seed_comps:
        print "walk seed comp %s " % name
        repos.walk( acs.compname2path(name) )
                    
    repos.print_selected()
        
    print "===> elevating %s owned folders to components or updating owners " % len(repos.select) 
    
    to_update = []
    to_insert = []
    to_delete = []
    
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
                to_update.append( c )
        except ResourceNotFound:
            c = Component( admin.env_open() )
            c.name = name
            c.owner = owner
            print "inserting component %s  " % ( c )
            to_insert.append(c)
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise


    print "===> make %s updates " % len(to_update)
    for c in to_update:
        c.update()
        
    print "===> make %s insertions " % len(to_insert)
    for c in to_insert:
        c.insert()
         
    print "===> purge auto components without corresponding repository paths  " 
    
    auto_comps = [ c.name for c in Component.select(admin.env_open()) if acs.is_autocomp(c.name) ]
    for name in auto_comps:
        path = acs.compname2path( name )
        node = repos.current_node( path )
        if node==None:
            print "orphaned component %s  path %s not in repository ... delete " % ( c , path )
            to_delete.append(c)
        else:
            print "component %s is valid " % c 
    
    print "===> make %s deletions " % len(to_delete)
    for c in to_delete:
        c.delete()





if __name__=='__main__':
    autocomp(sys.argv[1:])







