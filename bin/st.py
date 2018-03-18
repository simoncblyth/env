#!/usr/bin/env python
"""
Dump status of mercurial, svn, git repos in pwd 

"""
import os, commands, logging
log = logging.getLogger(__name__)

class Status(object):

    @classmethod
    def identify(cls, _):
        if os.path.isdir(os.path.join(_, ".hg")):
            typ = "hg"
        elif os.path.isdir(os.path.join(_, ".svn")):
            typ = "svn"
        elif os.path.isdir(os.path.join(_, ".git")):
            typ = "git"
        else:
            typ = None
        pass
        return typ
 
    @classmethod
    def command(cls, _):
        typ = cls.identify(_)
        if typ == "hg":
            cmd = "hg status %s " % _
        elif typ == "svn":
            cmd = "svn status %s " % _
        elif typ == "git":
            cmd = "git --work-tree=%(_)s --git-dir=%(_)s/.git status --porcelain " % locals()
        else:
            cmd = None
        pass
        return cmd

    def __init__(self, base):

        assert os.path.isdir(base)       

        typ = self.identify(base)  
        gls = commands.getoutput("git ls-files").split("\n") if typ == "git" else []
        gdi = set(map(os.path.dirname, gls))

        cmd =  self.command(base)

        log.debug(" typ %s cmd %s base %s " % (typ, cmd, base) ) 

        if not cmd is None:
            self.dump(cmd)
        pass

        self.typ = typ
        self.gls = gls
        self.gdi = gdi

      
        ls =  os.listdir(base)
        fls = filter(lambda _:os.path.isfile(_), ls)
        
        self.skipdir = []
        self.skipfile = filter(lambda _:not _ in gls, fls)

        for _ in filter(lambda _:os.path.isdir(_) and not os.path.islink(_), ls):
            cmd = self.command(_)
            if cmd is None: 
                if _ in self.gdi:
                    pass
                else:
                    self.skipdir.append(_)       
                pass
                continue
            pass
            self.dump(cmd)
        pass

    def dump(self, cmd):
        log.debug("dump: [%s] " % cmd )
        rc, out = commands.getstatusoutput(cmd)
        if len(out) > 0 or rc != 0:
            print "\n\n", cmd, "\n", out 
            #assert rc == 0 
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    st = Status(".")

    print "\n\n", "%d Dirs Skipped\n" % len(st.skipdir), "\n".join(st.skipdir)
    print "\n\n", "%d Files Skipped\n" % len(st.skipfile), "\n".join(st.skipfile)
    #print "\n\n", "%d Git Files\n" % len(st.gls), "\n".join(st.gls)
    #print "\n\n", "%d Git Dirs\n" % len(st.gdi), "\n".join(st.gdi)
        









