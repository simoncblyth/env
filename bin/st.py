#!/usr/bin/env python
"""
st.py
=======

Dump status of mercurial, svn or git repos in PWD
and lists any untracked(non-repository managed) 
dirs or files within the PWD.

When PWD is a git repo (eg with HOME/.git layout)
the list of untracked accounts for that.

::

    cd
    st.py --detail 1
    st.py -d1
    st.py -d2

    st.py --remotes-json 2>/dev/null  > .dotmeta/remotes.json


TODO
-----

* just give the TrackingRemote (where pushes will go) 
  rather than listing all unless extra detail is requested, 
  persist this into metadata : to pick which to clone from
  on recovery 

* status vs upstream for mercurial and git ? need to push ?
  but do not want to go out to network, as st.py 
  is used often and needs to stay local and fast


"""
import os, sys, commands, logging, argparse, json
from collections import OrderedDict
log = logging.getLogger(__name__)


class Remote(object):
    def __init__(self, cmd, typ, detail):
        meta = OrderedDict()
        lines = commands.getoutput(cmd).split("\n")

        if detail > 3:
            log.info("cmd : %s " % cmd )
            log.info("lines : %s " % "\n".join(lines) )
        pass
        for line in filter(None,lines): 
            if typ == "git":
                rem, url = self.git_parse(line) 
            elif typ == "hg":
                rem, url = self.hg_parse(line) 
            elif typ == "svn":
                rem, url = self.svn_parse(line) 
            else:
                assert 0 
            pass

            if rem in meta:
                assert meta[rem] == url
            else:
                meta[rem] = url
            pass
        pass
        self.meta = meta

    def git_parse(self, line):
        elem = line.split("\t")
        if len(elem) != 2:
            log.fatal("expecting two tabsep elem \"%s\" got %s " % ( line, repr(elem)))      
            assert 0 
        pass
        rem, val = elem
        url, bkt = val.split()
        return rem, url

    def hg_parse(self, line):
        rem, eq, url = line.split()
        assert eq == "="
        return rem, url

    def svn_parse(self, line):
        key, url = line.split()
        assert key == "URL:"
        return "URL", url

    def __str__(self):
        return "\n".join(["%10s : %s " % ( k, v) for k,v in self.meta.items() ])



class Repo(object):

    @classmethod
    def Identify(cls, _):
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
    def Status(cls, _):
        typ = cls.Identify(_)
        if typ == "hg":
            cmd = "hg status %s " % _
        elif typ == "svn":
            cmd = "svn status %s " % _
        elif typ == "git":
            #cmd = "git --work-tree=%(_)s --git-dir=%(_)s/.git status --porcelain " % locals()
            cmd = "cd %(_)s ; git status -s " % locals()
        else:
            cmd = None
        pass
        return cmd

    @classmethod
    def Remote(cls, _):
        typ = cls.Identify(_)
        if typ == "git":
            #cmd = "git --work-tree=%(_)s --git-dir=%(_)s/.git remote -v " % locals()
            cmd = "cd %(_)s ; git remote -v " % locals()
        elif typ == "hg":
            cmd = "cd %(_)s ; hg paths" % locals()
        elif typ == "svn":
            cmd = "cd %(_)s ; svn info | grep '^URL:' " % locals()
        else:
            cmd = None
        pass
        return cmd

    @classmethod
    def TrackingRemote(cls, _):
        typ = cls.Identify(_)
        if typ == "git":
            cmd = "cd %(_)s ; git rev-parse --abbrev-ref --symbolic-full-name @{u} "   ## eg uow/master
        else:
            cmd = None
        pass
        return cmd


    @classmethod
    def Make(cls, base, detail):
        typ = cls.Identify(base)
        repo = cls(base, typ, detail) if not typ is None else None 
        return repo 

    def __init__(self, base, typ, detail):
        self.base = base
        self.typ = typ
        self.status_command = self.Status(base)
        self.remote_command = self.Remote(base)

        self.status = commands.getoutput(self.status_command)
        self.remote = Remote(self.remote_command, typ, detail)

        gls = commands.getoutput("git ls-files").split("\n") if typ == "git" else []
        gdi = set(map(lambda _:_.split("/")[0], filter(lambda _:_.find("/") > -1, gls)))

        self.gls = gls
        self.gdi = gdi

    def __str__(self):  

        rem = str(self.remote).split("\n")
        sta = str(self.status).split("\n") 
        if len(rem) == 1:
            lines = ["## %20s : %s "  % (self.base, rem[0] ) ] 
        else:
            lines = ["## %20s : %s "  % (self.base, " ".join(rem)) ]  
        pass
        lines += [""]
        
        if len(self.status) > 0:
            lines += sta + ["",""]
        pass
        return "\n".join(lines)



class Untracked(object):
    def __init__(self, gls, gdi, ls, detail=0):
        self.gls = gls
        self.gdi = gdi
        self.detail = detail

        self.skipdir = []
        fls = filter(lambda _:os.path.isfile(_), ls)
        self.skipfile = filter(lambda _:not _ in gls, fls)
        self.skipdir = []

    def add(self, _):
        if _ in self.gdi:
            pass
        else:
            self.skipdir.append(_)       
        pass

    def __str__(self):
        lines = ["","", "%d Dirs Untracked" % len(self.skipdir), "" ] + self.skipdir + [""]
        lines += ["", "", "%d Files Untracked" % len(self.skipfile), ""] + self.skipfile + [""]
        if self.detail > 1:
            lines += ["", "", "%d Git Files" % len(self.gls), ""] + self.gls + [""] 
            lines += ["", "", "%d Git Dirs" % len(self.gdi), ""] + self.gdi + [""] 
        pass
        return "\n".join(lines)


class Home(object):

    @classmethod
    def parse_args(cls, **kwa):

        d = {}
        d["base"] = "."
        d["level"] = "INFO"
        d["detail"] = 0
        d["remotes-json"] = None
        d["links-json"] = None
        d["format"] = "%(asctime)-15s %(levelname)-7s %(name)-20s:%(lineno)-3d %(message)s"
        d.update(kwa)

        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('base', nargs="*", default=d["base"], help='base directory')
        parser.add_argument('--level', default=d["level"], help='log level')
        parser.add_argument('--remotes-json', default=d["remotes-json"], help='write repository remote metadata to path given or stdout if -')
        parser.add_argument('--links-json', default=d["links-json"], help='write link metadata to path given or stdout if -')
        parser.add_argument('-d', '--detail', type=int, default=d["detail"], help='detail level')
        
        args = parser.parse_args()
        logging.basicConfig(level=getattr(logging, args.level.upper()), format=d["format"])

        return args
 

    def __init__(self, args):

        self.args = args 
        base = args.base
        detail = args.detail
        assert os.path.isdir(base)       

        remo = OrderedDict()
        repos = []

        repo = Repo.Make(base, detail)

        if repo is not None:
            gls = repo.gls
            gdi = repo.gdi
            remo[base] = repo.remote.meta
            repos.append(repo)
            sys.stderr.write( str(repo))
        else:
            gls = []
            gdi = []
        pass

        ls =  os.listdir(base)
        other = Untracked(gls, gdi, ls)

        link = OrderedDict()
        for _ in filter(lambda _:os.path.islink(_), ls):
            link[_] = os.readlink(_)
        pass
        
        for _ in filter(lambda _:os.path.isdir(_) and not os.path.islink(_), ls):

            if detail > 1:
                log.info("checking dir %s " %  _ )
            pass
            repo = Repo.Make(_, detail)
            if repo is None: 
                other.add(_)
                continue
            pass
            remo[_] = repo.remote.meta
            repos.append(repo)
            sys.stderr.write( str(repo))
        pass
        sys.stderr.write( str(other))
        self.other = other
        self.remo = remo
        self.link = link
        self.repos = repos

    def out(self, meta, path, indent=None):
        if path == "-":
            fp = sys.stdout
        else:
            fp = file(path, "w")
        pass 
        json.dump(meta, fp, indent=indent)

    def __str__(self):
        return "\n".join(map(str, self.repos + [self.other]) ) 


if __name__ == '__main__':

    args = Home.parse_args()
    home = Home(args)

    if args.remotes_json is not None:
        home.out(home.remo, args.remotes_json, indent=3)
    pass
    if args.links_json is not None:
        home.out(home.link, args.links_json, indent=3)
    pass



       









