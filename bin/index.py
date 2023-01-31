#!/usr/bin/env python
"""
Usage::

    index.py 
    open http://localhost/env/presentation/index.html

OR::

    index.sh   # which does the above 

Change the descriptions by making changes to 
meta sections in the source .txt RST files
of form::

      .. meta:: 
         :description: (Jan 2020) Blah blah 


Default arguments are::

base
   ~/simoncblyth.bitbucket.io/env/presentation
srcbase
   ~/env/presentation



"""

from __future__ import print_function
import os, sys, logging, re, argparse, datetime
from lxml import etree
from dateutil.parser import parse
log = logging.getLogger(__name__)


moyrptn = re.compile("(?P<month>[a-z]*)(?P<year>\d{4})")

ymdptn = re.compile("(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})")


def dt_to_epo(dt):
    return int(dt.strftime("%s"))

def extract_date_from_moyr(txt):
    m = moyrptn.search(txt)
    if not m: return None
    moyr = m.string[m.start():m.end()]  
    if moyr in ["2021","2020"]:
        return None  
    pass      
    dt = parse(moyr, default=None)    
    return dict(dt=dt, src="extract_date_from_moyr", moyr=moyr) 

def extract_date_from_ymd(txt):
    m = ymdptn.search(txt)
    if not m: return None
    ymd = m.string[m.start():m.end()]  
    dt = datetime.datetime.strptime(ymd, "%Y%m%d")    
    return dict(dt=dt, src="extract_date_from_ymd") 

def extract_date_from_name(name):
    dt = extract_date_from_ymd(name)
    if dt is None:
       dt = extract_date_from_moyr(name)
    pass
    return dt 


bptn = re.compile("\((?P<inbrk>.*)\)")
def extract_date_from_desc(desc):
    bm = bptn.search(desc)
    if not bm: return 0
    d = bm.groupdict()
    txt = "%(inbrk)s" % d  
    dt = parse(txt, default=None)
    return dict(dt=dt, src="extract_date_from_desc" )


def extract_date(name, desc):
    dt = None
    if dt is None:
       dt = extract_date_from_name(name)
    pass
    if dt is None:
       dt = extract_date_from_desc(desc) 
    pass
    if dt is None or dt is 0:
       dt = dict(dt=datetime.datetime.now(), src="extract_date_default")
    pass
    return dt 



class Doc(object):
    def __init__(self, path):
        if os.path.exists(path):
            meta = self.Meta(path)
        else:
            meta = {}
        pass
        self.meta = meta
        self.path = path 
        self.desc = meta.get("description", "-")
    def __repr__(self):
        return "%100s : %s " % ( self.path, self.desc )


class Html(Doc):
    @classmethod
    def Meta(cls, path):
        """
        :param path:
        :return d:
        """
        tree = etree.parse( open(path),  etree.HTMLParser() )
        root = tree.getroot()
        d = {}
        for meta in root.xpath("./head/meta[@name and @content]"): 
            name = meta.xpath("@name")[0]
            content = meta.xpath("@content")[0]
            d[name] = content
        pass
        return d

class Rst(Doc):
    @classmethod
    def Meta(cls, path):
        """
        :param path:
        :return d:
        """
        lines = list(map(str.strip,open(path).readlines()))
        ptn = re.compile(":(?P<key>\S*): (?P<val>.*)$")
        d = {}
        META = ".. meta::"
        if not META in lines:
            return d
        pass
        imeta = lines.index(META)
        iblank = imeta + lines[imeta:].index("")
        for line in lines[imeta+1:iblank]:
            #print(line) 
            m = ptn.match(line)
            assert m, ("bad line", line)
            gd = m.groupdict()
            key = gd["key"]
            val = gd["val"]
            d[key] = val 
            pass
        pass
        #log.info("path %s lines %s d %s " % (path, len(lines), d))
        return d 



class Item(object):
    def __init__(self, name, idx):
        """
        :param name: 
        :param idx: Index instance

        Collects metadata from both derived html presentation and 
        source RST 

        """
        #log.info(name)
        html = Html(idx.htmlpath(name))
        rst = Rst(idx.rstpath(name))

        if "20210208" in name:
            print(html)
            print(rst)
        pass


        meta = {}
        meta.update(html.meta)
        meta.update(rst.meta)
        desc = meta.get("description", "")

        dtd = extract_date(name, desc) 
        dt = dtd["dt"]
        src = dtd["src"]
        moyr = dtd.get("moyr","-")      
 
        #log.info("name %s desc %s dt %s src %s " % (name, desc, dt, src))

        epo = dt_to_epo(dt)

        self.meta = meta 
        self.desc = desc
        self.name = name
        self.dt = dt
        self.epo = epo
        self.src = src
        self.moyr = moyr

    def _get_lines(self):
        l = []
        l.append("<li> <a href=\"%s.html\"> %s.html </a>   <a href=\"%s.txt\">.txt</a> %s </li>" % (self.name, self.name, self.name,self.desc))
        return l
    lines = property(_get_lines)

 
    def __str__(self):
        return "\n".join(self.lines) 

    def __repr__(self):
        return "%80s epo %15s dt %s src %s moyr %s " % (self.name, self.epo, self.dt.strftime("%c"), self.src, self.moyr ) 

class Index(object):
    def __init__(self, base, srcbase):
        """
        :param base: directory containing .html presentations
        :param srcbase: directory containing .txt RST sources of the presentations

        1. Lists .html in base and extracts names 
        2. creates items list 

        Ordering is by item.date

        """
        log.info("base %s srcbase %s " % (base, srcbase))
        base = os.path.expandvars(os.path.expanduser(base))
        assert os.path.isdir(base), base
        srcbase = os.path.expandvars(os.path.expanduser(srcbase))
        assert os.path.isdir(srcbase), srcbase
        pass
        self.base = base 
        self.srcbase = srcbase 

        exclude_html = ["index.html", "test.html"]

        names = list(filter(lambda p:p.endswith(".html") and not p in exclude_html,os.listdir(self.base)))
        names = list(map(lambda n:n[:-5], names))  # remove .html
        items = list(map(lambda name:Item(name, self), names ))
        items = sorted(items, key=lambda item:item.epo, reverse=True)
        for item in items:
            print(repr(item))
        pass 

        self.items = items

    def htmlpath(self, name):
        return os.path.join(self.base, "%s.html" % name)
    def rstpath(self, name):
        return os.path.join(self.srcbase, "%s.txt" % name)

    def write(self):
        path = os.path.join(self.base, "index.html")
        if os.path.exists(path):
            os.remove(path) 
        pass 
        log.info("writing to %s " % path)
        out = open(path, "w") 
        print(str(self), file=out) 

    def _get_lines(self):
        l = []
        l.append("<html>")
        l.append("<p> generated by ~/env/bin/index.sh/index.py : insert a date in brackets into description metadata to control item order if order not detected from name</p>")
        l.append("<p> see also : <a href=\"https://simoncblyth.bitbucket.io\" > https://simoncblyth.bitbucket.io </a> ")
        l.append("<!-- open http://localhost/env/presentation/index.html -->")
        l.append("<ul>")
        for item in self.items:
            l.extend(item.lines)
        pass
        l.append("</ul>")
        l.append("</html>")
        return l

    lines = property(_get_lines)
 
    def __str__(self):
        return "\n".join(self.lines) 




def parse_args(doc):
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)

    d = {}
    d["level"] = "INFO" 
    d["base"] = "~/simoncblyth.bitbucket.io/env/presentation" 
    d["srcbase"] = "~/env/presentation" 
    d["format"] = "%(asctime)-15s %(levelname)-7s %(name)-20s:%(lineno)-3d %(message)s"

    parser.add_argument('--base', default=d["base"], help='base directory. Default: %(default)s ')
    parser.add_argument('--srcbase', default=d["srcbase"], help='srcbase directory. Default: %(default)s')
    parser.add_argument('--level', default=d["level"], help='log level. Default: %(default)s')
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.level.upper()), format=d["format"])
    return args

if __name__ == '__main__':

    args = parse_args(__doc__)
    idx = Index(args.base, args.srcbase)
    idx.write()



