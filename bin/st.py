#!/usr/bin/env python
"""
Dump status of mercurial, svn, git repos in pwd 

"""
import os, commands

class Status(object):

    @classmethod
    def command(cls, _):
        if os.path.isdir(os.path.join(_, ".hg")):
            cmd = "hg status %s " % _
        elif os.path.isdir(os.path.join(_, ".svn")):
            cmd = "svn status %s " % _
        elif os.path.isdir(os.path.join(_, ".git")):
            cmd = "git --work-tree=%(_)s --git-dir=%(_)s/.git status " % locals()
        else:
            cmd = None
        pass
        return cmd

    def __init__(self, base):
        self.skipdir = []
        ls =  os.listdir(base)

        self.skipfile = filter(lambda _:os.path.isfile(_), ls)

        for _ in filter(lambda _:os.path.isdir(_) and not os.path.islink(_), ls):
            cmd = self.command(_)
            if cmd is None:
                self.skipdir.append(_)       
                continue
            pass
            self.dump(cmd)
        pass

    def dump(self, cmd):
        rc, out = commands.getstatusoutput(cmd)
        if len(out) > 0 or rc != 0:
            print "\n\n", cmd, "\n", out 
            #assert rc == 0 
        pass


if __name__ == '__main__':
    st = Status(".")

    print "\n\n", "%d Dirs Skipped\n" % len(st.skipdir), "\n".join(st.skipdir)
    print "\n\n", "%d Files Skipped\n" % len(st.skipfile), "\n".join(st.skipfile)
        









