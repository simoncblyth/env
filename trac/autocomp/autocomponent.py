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
               trac-; autocomp-; autocomp-sudosync <name-defaulting-to-TRAC_INSTANCE>  
           
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
        for s_path,s_owner in self.select:
            if s_path == path:
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
                name = self.schema.path2compname(pp)
                pnode = self.current_node(pp) 
                if self.schema.is_seedcomp(name):
                    print "%s parent_select [%s] [%s] stop selection above this seed node " % ( self, pp , name ) 
                    return
                if pnode.isdir:
                    self.select_node( pnode )
    
    
    def select_node(self, node):
        props = node.get_properties()
        self.select.append( (node.path , self.schema.prop_get( props ) ))
    
    def _walk(self, node):
        """ recursive tree walker ... sticking to directories and not following branches or tags 
            and making a selection of directory nodes that have an OWNER property 
        """
        if not(node):
            return
            
        if node.name in ['branches','tags']:
            return
        if node.isdir:
            #print "_walk %s " % node.path
            props = node.get_properties()
            if self.schema.prop_select( props ):
                self.select_node( node )
                self.parent_select(node.path)
        
        if node.isdir:
            for n in node.get_entries():
                self._walk(n)

    def print_selected(self):
        for path,owner in self.select:
            print "%s:%s" % (path, owner )





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
        print "===> use seed components to direct the repository walk, selecting owned nodes .. and their parents "
        seed_comps = [ c.name for c in Component.select(admin.env_open()) if acs.is_seedcomp(c.name) ]
    else:
        acs = BaseTrunkAutoComponent( "owner" , "blyth" , "//" )
        seed_comps = ["//"] 
    print acs
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
    
    for path,owner in repos.select:
        name = acs.path2compname(path)
        match_comps = [ (c.name,c.owner) for c in Component.select(admin.env_open()) if c.name == name ]        
        nmatch = len(match_comps)

        msg = ""
        if nmatch == 0 :
            to_insert.append( (name,owner) )
            msg = "insert"    
        elif nmatch == 1 :
            c_name, c_owner = match_comps[0]
            if c_owner == owner:
                msg = "same owner -- no action" 
            else:
                msg = "update former owner %s " % ( c_owner ) 
                to_update.append( (name, owner) )
        else:
            msg = "ERROR to many matching components "
       
        print "[%-3s] %-15s %-100s %-100s %s " % ( nmatch, owner , path , name , msg )                      
        

    print "===> make %s updates " % len(to_update)
    for name, owner in to_update:
        admin._do_component_set_owner( name, owner )
        
    print "===> make %s insertions " % len(to_insert)
    for name, owner in to_insert:
        admin._do_component_add( name, owner )
         
    print "===> purge auto components without corresponding repository paths  " 
    
    auto_comps = [ c.name for c in Component.select(admin.env_open()) if acs.is_autocomp(c.name) ]
    for name in auto_comps:
        path = acs.compname2path( name )
        if not(repos.repos.has_node(path)):
            to_delete.append(name)
      
    print "===> make %s deletions " % len(to_delete)
    for name in to_delete:
        admin._do_component_remove( name )





if __name__=='__main__':
    autocomp(sys.argv[1:])







