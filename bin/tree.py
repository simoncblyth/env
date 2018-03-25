#!/usr/bin/env python
"""
tree.py 
=========

Metadata digests tree, enabling quick determination if 
a binary from one folder also exists in another without depending
on the file name.

For example running the below will find files with names 
matching a set of patterns (eg "*.pdf") 
that are in both directory trees, based on file content 
(NOT the file name)::

    tree.py ~/tree ~/Downloads

For ipython running::

   run env/bin/tree.py ~/tree ~/Downloads

TODO:

* currently no mechanism to cope with updates to
  the directory content, have to manually delete the
  metadata file to force recreation for directories
  to update::

      rm .tree_pdf.json


"""
import sys, os, hashlib, json, logging, argparse
from collections import OrderedDict
from fnmatch import fnmatchcase

log = logging.getLogger(__name__)

class Path(dict):

    @classmethod
    def digest_(cls, path):
        """ 
        :param path:
        :return: md5 hexdigest (matching /sbin/md5) of path content or None if non-existing path 
        """
        if not os.path.exists(path):return None
        if os.path.isdir(path):return None
        md5 = hashlib.md5()
        with open(path,'rb') as f:  
            for chunk in iter(lambda: f.read(8192),''): 
                md5.update(chunk)
            pass
        pass
        return md5.hexdigest()

    @classmethod
    def is_matched(cls, name, pattern_list ):
        return len(filter(lambda pattern:fnmatchcase(name,pattern), pattern_list)) > 0 

    @classmethod
    def Make(cls, path, root):

        assert os.path.exists(path)
        assert path.startswith(root)
        relpath = path[len(root)+1:]
 
        st = os.stat(path)
        kwa = {}
        kwa["mode"] = st.st_mode
        kwa["size"] = st.st_size
        kwa["atime"] = st.st_atime
        kwa["mtime"] = st.st_mtime
        kwa["ctime"] = st.st_ctime
        kwa["digest"] = cls.digest_(path)
        kwa["path"] = path
        kwa["relpath"] = relpath

        return cls(**kwa)
    
    def __init__(self, *args, **kwa):
        dict.__init__(self, *args, **kwa)
           
    FMT = "%(size)10s %(digest)s %(relpath)s "
    def __str__(self):
        return self.FMT % self


class Tree(object):

    @classmethod
    def parse_args(cls, **kwa):

        d = {}
        d["base"] = "."
        d["level"] = "INFO"
        d["ftyp"] = "pdf"
        d["label"] = "pdf"
        d["detail"] = 0
        d["format"] = "%(asctime)-15s %(levelname)-7s %(name)-20s:%(lineno)-3d %(message)s"
        d.update(kwa)

        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('base', nargs="+", default=d["base"], help='base directory')
        parser.add_argument('--ftyp', default=d["ftyp"], help='comma delimited list of file types')
        parser.add_argument('--label', default=d["label"], help='identifying label used for metadata json file')
        parser.add_argument('--level', default=d["level"], help='log level')
        parser.add_argument('-d', '--detail', type=int, default=d["detail"], help='detail level')
        
        args = parser.parse_args()
        logging.basicConfig(level=getattr(logging, args.level.upper()), format=d["format"])

        args.patterns = map(lambda _:"*.%s" % _, args.ftyp.split(","))
        return args
 

    @classmethod 
    def JSPath(cls, base, label):
        return os.path.join(base, ".tree_%s.json" % label)

    @classmethod 
    def MakeOrLoad(cls, base, patterns=["*.pdf"], label="pdf"):

        log.info( " base:%s patterns:%r label:%s " % (base, patterns, label))
        jspath = cls.JSPath(base, label)
        if not os.path.exists(jspath):
            tr = cls(base, patterns, label) 
            tr.recurse()
            tr.save()
        else:
            tr = cls(base, patterns, label) 
            tr.load()
        pass
        return tr
         

    def __init__(self, base, patterns=[], label="nolabel"):
        self.meta = OrderedDict()
        self.base = os.path.abspath(os.path.expanduser(os.path.expandvars(base)))
        self.patterns = patterns
        self.label = label
        
    def recurse(self):
        self.recurse_(self.base, 0 ) 

    def recurse_(self, fold, depth ):
        names = os.listdir(fold)
        for name in names:
            path = os.path.join(fold, name)
            if os.path.isdir(path):
                 self.recurse_(path, depth+1)
            else:
                 if Path.is_matched(name, self.patterns): 
                     pd = Path.Make(path, self.base)
                     pd["idx"] = len(self.meta)
                     print pd
                     self.meta[pd["digest"]] = pd
                 pass
            pass
        pass

    def save(self, indent=3):
        path = self.JSPath(self.base, self.label)
        log.info("saving to %s " % path ) 
        json.dump(self.meta, file(path,"w"), indent=indent)

    def load(self): 
        path = self.JSPath(self.base, self.label)
        log.info("loading from %s " % path ) 
        s = file(path, "r").read()
        meta = json.loads(s)
        meta = OrderedDict(sorted(meta.items(), key=lambda kv:kv[1]["idx"]))  
        pass
        for k, v in meta.items():
            self.meta[k] = Path(v)
        pass

    def __repr__(self):
        return "Tree %4s : %s " % ( len(self.meta), self.base )
    def __str__(self):
        return "\n".join( ["", repr(self), ""] + map(lambda kv:Path.FMT % kv[1], self.meta.items() ))

    def __len__(self):
        return len(self.meta)
    def __getitem__(self, i):
        k = self.getkey(i)
        return self.meta[k]
    def getkey(self, i):
        return self.meta.keys()[i]
 

    @classmethod 
    def Compare(cls, a, b):
        sa = set(a.meta.keys())
        sb = set(b.meta.keys()) 
        return  
        

if __name__ == '__main__':

    args = Tree.parse_args()
    nbase = len(args.base)

    if nbase == 1:
        ta = Tree.MakeOrLoad(args.base[0], args.patterns, args.label  )
        print ta
    elif nbase == 2:
        ta = Tree.MakeOrLoad(args.base[0], args.patterns, args.label  )
        tb = Tree.MakeOrLoad(args.base[1], args.patterns, args.label  )

        print ta
        print tb

        print 
        print "in both"
        print
        for b in tb:
            if b['digest'] in ta.meta:
                a = ta.meta[b['digest']]
                print " b:%s a:%s " % ( str(b), str(a) )
            pass
        pass


    else:
        assert 0, "expecting 1 or 2 bases only "





