#!/usr/bin/env python
"""
Getting PDFs and other binaries out of a source tree and into
a parallel hierarcy, typically for apache presentation from there.
The veracity of the copy is checked via digests.  

This script just looks and suggests the commands to run, it does not do anything.  
To act on suggestions pipe the output to the shell.

Although Sphinx has a download directive, it works via duplication

Usage, takes 2 runs to 1st copy binaries from working copy and then to verify them and delete in working copy

   cd ~/workflow
   ./sweep.py           ## check the copies/deletions to be done
   ./sweep.py | sh 	## do them
   ./sweep.py           ## check the copies/deletions to be done
   ./sweep.py | sh 	## do them 

   cd ~/env
   
Note that this module is imported by these lightweight sweep scripts that live
in the root of each svn repository.

Refer to resources with urls like:

  * http://localhost/edocs/whatever.pdf	
  * http://localhost/wdocs/notes/dev/whatever.pdf	
  * OR :w:`notes/dev/whatever.pdf`

TODO:

#. hmm, I now want the working copy to live on G4PB for easy access to notes : need to change names from workflow to wdocs
#. could base off the ``svn status`` output rather than walking and calling ``svn status`` for each 

   #. ` svn st | grep ^\?`


"""
from __future__ import with_statement
import os, hashlib, logging, re
log = logging.getLogger(__name__)

def digest_(path):
    """
    :param path:
    :return: md5 hexdigest of the content of the path or None if non-existing path

    http://stackoverflow.com/questions/1131220/get-md5-hash-of-a-files-without-open-it-in-python

    Confirmed to give same hexdigest as commandline /sbin/md5::

	    md5 /Users/blyth/workflow/notes/php/property/colliers-4q2011.pdf 
	    MD5 (/Users/blyth/workflow/notes/php/property/colliers-4q2011.pdf) = 3a63b5232ff7cb6fa6a7c241050ceeed

    """
    if not os.path.exists(path):return None
    if os.path.isdir(path):return None
    md5 = hashlib.md5()
    with open(path,'rb') as f: 
	for chunk in iter(lambda: f.read(8192),''): 
	    md5.update(chunk)
    return md5.hexdigest()

def svnstatus_(path):
    """
    :param path:
    :return: svn status string of the path or "-" if blank 
    """
    st = os.popen("svn status %s 2>/dev/null" % path).read()
    if len(st) == 0:
        return "-"
    else:
        return st[0]

class DPath(object):
    """
    Holder object for:

    #. path
    #. content digest
    #. svn status string

    """
    def __init__(self, path, svnst=None):
	"""
	:param path: 
	:param svnst: status string
	"""
	self.path = path 
        self.digest = digest_(path)
	if svnst:
            self.svnst = svnst
        else:    
	    self.svnst = svnstatus_(path)
    def __repr__(self):
	return "[%s] %s %s" % ( self.svnst, self.path, self.digest )     


class Sweeper(list):
    """
    Keep working copy clean by sweeping chosen filetypes into a 
    parallel heirarchy.

    Deficiencies:

    #. probably will not handle major changes in working copy structure very gracefully

       * eg does not delete at target on changing source directory names, so orphans will result

    """
    def __init__(self, src, tgt, exts='.pdf .txt .html .xml'):	
	"""
	:param src: source directory
	:param tgt: target directory
	:param exts: filetype to be sweeped if they have svn status of **?**
	:params skipd: directory names to not recurse into, in new walk this is not needed 
	"""
	xx_ = lambda _:os.path.expandvars(os.path.expanduser(_))
        self.src = xx_(src)
	self.tgt = xx_(tgt)
	self.exts = exts.split()
        self.cmds = []
        self.walk()

    def walk(self):
	"""
	Pattern match the `svn status` of the `src` directory  passing all matches to `handle`
	Note that only a single `svn status` is required, so much more efficient compared to oldwalk.
	"""
	cmd = "svn status %s " % self.src
	ptn = re.compile("^(\S)\s*(\S?)\s*(%s.*)\s*$" % self.src)
	for line in os.popen(cmd).readlines():
            m = ptn.match(line)
            if m:
		 groups = m.groups()
		 assert len(groups) == 3 
		 stat_,atat_,path = groups
		 stat = stat_.rstrip()
		 log.debug( "[%s][%s] %s " % ( stat, atat_, path ))
		 self.handle( path, stat )
            else:
		 log.info("no match %s " % line )   

    def oldwalk(self):
	"""
	Old way used ordinary os.walk and then `svn status` separately for every path, 
	requiring separate svn status checks for each path within `handle`
	"""
        for dirpath, dirs, names in os.walk(self.src):
	    rdir = dirpath[len(self.src)+1:]	
            #for skp in self.skipd:		 
	    #    if skp in dirs:
	    #        dirs.remove(skp)  
	    for name in names:
                root, ext = os.path.splitext(name)
                if not ext in self.exts: 
                    continue
                spath = os.path.join(dirpath, name)
                self.handle(spath)

    def handle(self, spath, svnst=None):
	"""

        Considers a source path and appends shell commands to either:

	#. create target directory if not created and copy the binary littering source to target
	#. remove the littering binary from working copy source if digests of verify a good copy to target 

	:param spath: full path 
	:param svnst: svn status string
	"""
	rpath = spath[len(self.src)+1:]
	sp = DPath(spath, svnst=svnst) 
        if sp.svnst != '?':
	    return 
        tpath = os.path.join(self.tgt,rpath)
	tp = DPath(tpath, svnst="-") 
	log.info("sp %s" % sp )
	log.info("tp %s" % tp )
        if sp.digest == tp.digest:
	    cmd = "rm -f \"%s\" " % ( spath ) 	    
	else:   
	    cmd = "mkdir -p \"%s\" && cp \"%s\" \"%s\" " % ( os.path.dirname(tpath), spath, tpath )
        self.cmds.append(cmd)

    def __repr__(self):
        return "\n".join(self.cmds)
	    

if __name__ == '__main__':
    pass


