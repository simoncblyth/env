import sys
import trac.env
from trac.versioncontrol.api import NoSuchNode, NoSuchChangeset

"""

   See /Users/blyth/trac/trac/versioncontrol/web_ui/browser.py
   for how the browser module accesses the repository ..

   This could be used to auto populate a component list 
   
      can easily recurse over the full tree   
     
     ... tis an acceptable config burden to start with a list of 
       top level paths :  /dybgaudi  /gaudi 
     
     ... the main difficulty is how to prune the tree 
     ... contant depth cuts will not work
            dybgaudi/trunk/DybTest
            dybgaudi/trunk/Simulation/ElecSim 

        "trunk" need to be walked but then skipped in component name
               

    can control where to walk, via svn property 
      ... ie only walk into folders with an "owner" property defined
      ... stop walking 


   TESTING ON G WITH RECOVERED FROM BACKUP REPOSITORIES :   
        sudo python $ENV_HOME/trac/script/access.py /var/scm/tracs/dyw

"""



class entry(object):
    __slots__ = 'name rev kind isdir path content_length'.split()
    def __init__(self, node):
        for f in entry.__slots__:
            setattr(self, f, getattr(node, f))
    def __repr__(self):
        return "<entry name %s path %s rev %s kind %s isdir %s   >" % ( self.name, self.path , self.rev, self.kind , self.isdir  )
                
                                        
def main():
    """
        Scripted access to a trac environment with recursive walking around the repository 
    """
    
    if len(sys.argv) != 2:
        print "Usage: %s path_to_trac_environment" % sys.argv[0]
        sys.exit(1)
    tracdir = sys.argv[1]
    trac_env = trac.env.open_environment(tracdir)

    authname = "blyth"
    repos = trac_env.get_repository(authname)
    print repos
    repowalk( "/" , repos )


def repowalk( path , repos ):
    rev = None   ## means the latest    
    node = None
    try:
        if rev:
            rev = repos.normalize_rev(rev)
        rev_or_latest = rev or repos.youngest_rev
        node =  repos.get_node(path, rev_or_latest ) 
    except NoSuchChangeset, e:
        raise ResourceNotFound(e.message, _('Invalid Changeset Number'))
    walk(node)
  

def walk( node ):
    if not(node):
        return
    enode=entry(node)
    if node.isdir:
        print enode
  
    if node.isdir:
        for n in node.get_entries():
            walk(n)
        
        
        






if __name__ == '__main__':
    main()

