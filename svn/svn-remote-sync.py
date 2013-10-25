
"""
[root@dayabay svn]# LD_LIBRARY_PATH=~blyth/local/python/Python-2.5.6/lib:$LD_LIBRARY_PATH ~blyth/local/python/Python-2.5.6/bin/python svn-remote-sync.py 
"""
import os
import logging
import xml.dom.minidom

log = logging.getLogger(__name__)

def get_rev(url):
    f = os.popen("svn log %s --limit 1 --xml"%url) 
    txt = f.read()
    dom = xml.dom.minidom.parseString(txt)

    rev = None

    ts = dom.getElementsByTagName("logentry")
    if len(ts) == 1:
        tag = ts[0]
        rev = tag.getAttribute("revision")

    return rev

def init_repo(src, dst):
    """
    :param src: eg. dayabay http://dayabay.ihep.ac.cn/svn/dybsvn
    :param dst: eg. dayabay1 http://202.122.39.101/svn/dybsvn
    """
    print os.popen("svnsync initialize %s %s"%(dst, src)).read()

def sync_repo(dst):
    """
    :param dst: eg. dayabay1 http://202.122.39.101/svn/dybsvn
    """
    log.info(os.popen("svnsync synchronize %s"%dst).read())
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    url = "http://dayabay.ihep.ac.cn/svn/dybsvn/dybgaudi/trunk"
    print get_rev(url)
    src = "http://dayabay.ihep.ac.cn/svn/dybaux"
    dst = "http://202.122.39.101/svn/dybaux"
    init_repo(src,dst)
    sync_repo(dst)
