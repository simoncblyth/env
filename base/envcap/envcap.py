#!/usr/bin/env python
"""
Usage via the python script directly is recommended as other
less direct methods (such as via shim bash or python scripts) 
are prone to tampering with the environment being captured::

    $HOME/env/base/envcap/envcap.py -tzero save   # capture zero env from start of .bash_profile
    $HOME/env/base/envcap/envcap.py -tpost save   # capture post env from end of .bash_profile

    $HOME/env/base/envcap/envcap.py -tcsa  save   

            # after doing some time consuming env setup cache to csa.json

    $HOME/env/base/envcap/envcap.py -zpost -tcsa cachediff

            # now cache the env difference between the "post.json" and "csa.json" 
            # dumps into a bash envcache script "csa.sh" 
            # precise paths controlled via template envvars

"""
import collections
import os, json, logging, time
log = logging.getLogger(__name__)

def mtime(path):
    return time.ctime(os.path.getmtime(path))

class Env(collections.OrderedDict):
    _skips = r"""_ HOME PWD OLDPWD LOGNAME SHELL SHLVL MAIL USER 
    EDITOR TERM DISPLAY SSH_TTY HOSTNAME PS1 SSH_INFOFILE SSH_CLIENT 
    SSH_CONNECTION LS_COLORS SSH_AGENT_PID SSH_AUTH_SOCK LC_ALL LC_CTYPE
    """

    @classmethod
    def skips(cls):
        return filter(None,cls._skips.replace("\n"," ").split(" "))

    def __init__(self, *args, **kwa):
        collections.OrderedDict.__init__(self, *args, **kwa)

    @classmethod
    def capture(cls):
        skips = cls.skips()
        kvs = filter(lambda kv:kv[0] not in skips,sorted(os.environ.items(),key=lambda kv:kv[0]))
        return cls(kvs)

    def save(self, path):
        log.info("save to %s " % path )
        with open(path,"w") as fp:
            json.dump(self, fp)

    def subtract(self, path):
        other = Env.load(path)

    @classmethod
    def load(cls, path):
        with open(path,"r") as fp:
            js = fp.read()
        return json.loads(js, object_hook=cls)

    @classmethod
    def diff(cls, zpath, path):
        pass
        log.info("diff zpath %s %s path %s %s " % (zpath,mtime(zpath), path,mtime(path)))

        c = cls()
        a = cls.load(zpath)
        b = cls.load(path)

        ak = set(a.keys())
        bk = set(b.keys())

        for k in sorted(ak.union(bk)):
            if k in ak and k in bk:
                if a[k] == b[k]:
                    log.debug("a == b : %s : %s == %s  " % (k,a[k], b[k]))
                else:
                    log.info("a != b : %s : %s != %s  " % (k,a[k], b[k]))
                    c[k] = b[k] 
                pass

            elif k in ak:
                log.info("a only: %s : %s " % (k,a[k]))
            elif k in bk:
                log.info("b only: %s : %s " % (k,b[k]))
                c[k] = b[k] 
            pass
        pass
        c['ENVCAP_META_ZPATH'] = zpath
        c['ENVCAP_META_ZPATH_MTIME'] = mtime(zpath)
        c['ENVCAP_META_PATH'] = path
        c['ENVCAP_META_PATH_MTIME'] = mtime(path)
        return c

    def __str__(self):
        return "\n".join(map(lambda kv:"export %s='%s'" % kv ,sorted(self.items())))
          



def parse_args(doc):
    from optparse import OptionParser
    op = OptionParser(usage=doc)

    envcap_json_tmpl = os.environ.get('ENVCAP_JSON_TMPL', "$HOME/.env/envcap/%s.json" ) 
    envcap_bash_tmpl = os.environ.get('ENVCAP_BASH_TMPL', "$HOME/.env/envcap/%s.sh" ) 

    op.add_option("-l", "--loglevel",   default="INFO", help="logging level : INFO, WARN, DEBUG. Default %default"  )
    op.add_option("--logformat", default="%(asctime)s %(name)s %(levelname)-8s %(message)s" , help="logging format" )
    op.add_option("-t", "--tag",       default="post", help="tag used for naming envcap files" )
    op.add_option("-z", "--zero",      default="zero",  help="separate tag used subtraction between environments" )
    op.add_option(  "--jsontmpl",      default=envcap_json_tmpl , help="Path template to save envcap json" )
    op.add_option(  "--bashtmpl",      default=envcap_bash_tmpl , help="Path template to save envcap bash files" )

    opts, args = op.parse_args()
    level = getattr( logging, opts.loglevel.upper() )
    logging.basicConfig(format=opts.logformat,level=level)
    return opts, args



def prep(tmpl, tag ):
    path = os.path.expandvars(os.path.expanduser(tmpl % tag ))
    dirp = os.path.dirname(path)
    if not os.path.exists(dirp):
        os.makedirs(dirp)
    pass
    return path


class Config(object):
    def __init__(self):
        opts, args = parse_args(__doc__)
        self.opts = opts
        self.args = args

    bash  = property(lambda self:prep(self.opts.bashtmpl,self.opts.tag))
    path  = property(lambda self:prep(self.opts.jsontmpl,self.opts.tag))
    zpath = property(lambda self:prep(self.opts.jsontmpl,self.opts.zero))


def main():
    cfg = Config()
    for cmd in cfg.args:
        if cmd == "save":
            e = Env.capture()
            e.save(cfg.path) 
        elif cmd == "diff":
            e = Env.diff(cfg.zpath, cfg.path) 
            print e
        elif cmd == "cachediff":
            e = Env.diff(cfg.zpath, cfg.path) 
            sh = cfg.bash
            log.info("cachediff writing to %s " % sh )
            with open(sh, "w") as fp:
                fp.write(str(e))
        else:
            log.warn("cmd % ignored " % cmd )


if __name__ == '__main__':
    main()


