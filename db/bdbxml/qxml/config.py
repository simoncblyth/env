#!/usr/bin/env python
"""

In [19]: dict([k,cfg.items(k)] for k in cfg.sections())
Out[19]: 
	{'containers': [('sys', '/tmp/hfagc/hfagc_system.dbxml'),
	                ('hfc', '/tmp/hfagc/hfagc.dbxml')],
'defaults': [('baseuri', ''),
('collection', 'dbxml:////tmp/hfagc/hfagc.dbxml')],
'namespaces': [('rez', '"http://hfag.phys.ntu.edu.tw/hfagc/rez"')]}

"""

from ConfigParser import ConfigParser

def read_config( path ):
    cfg = ConfigParser()
    cfg.readfp(open(path),"r")
    return dict([k,cfg.items(k)] for k in cfg.sections())

if __name__ == '__main__':
    cfg = read_config("/Users/blyth/env/db/bdbxml/qxml/hfagc.ini")

    for s in "defaults containers namespaces".split():
	 print "*%s*" % s   
	 for k,v in cfg.get(s,[]):    
	    print k,v     





