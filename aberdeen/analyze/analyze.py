#!/usr/bin/env python
"""
"""
import os, re, logging
from pprint import pformat
log = logging.getLogger(__name__)

pptn = re.compile("(?P<dir>.*?)run(?P<run>\d{5}).root$")


def analyze_(dir="",run=None,batchdir="/tmp/batch",tmpl="ana00000.sh.tmpl"):

    tmpl_ = open(tmpl,"r").read()     
    text = tmpl_ % locals()

    name = "ana%(run)s" % locals()
    path = os.path.join(batchdir,"script",name)
    sdir = os.path.dirname(path)  
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    sh = path + ".sh"
    log.info("writing %s for run %s " % (sh,run) )
    fp = open(sh,"w")
    fp.write(text)
    fp.close()

    cmd = "chmod ugo+x %(sh)s ; batch -f %(sh)s now " % locals()
    log.info("submitting %s " % cmd )
    fd = os.popen(cmd)
    for line in fd.readlines():
        log.info(line)   

def analyze(path):
    log.info(path)
    m = pptn.match(path)
    if not m:
        log.warn("path %s not matched" % path )  
    else:
        d = m.groupdict()
        if len(d['dir']) > 0 and d['dir'][-1] == "/":d['dir'] = d['dir'][:-1]
        log.info("matched path %s %s " % ( path, pformat(d) ))
        analyze_(**d)

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    analyze(sys.argv[1])
