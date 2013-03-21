#!/usr/bin/env python
"""

"""
import os, logging, re
from pprint import pformat
from optparse import OptionParser
log = logging.getLogger(__name__)
import re

def remove_droppings():
    """
    Suspect the need for this to clean up the __db.001
    indicates are missing some memory cleanup ?

    Nope : seems that the environment dir was not being controlled so PWD was used
    """
    import os, glob	
    files = glob.glob("__db.*")
    for file in files:
        os.remove(file)

def parse_args():
    """
    Try to duplicate the boost_program_options interface in C++ qxml
    """
    parser = OptionParser()
    parser.add_option("-l", "--level",  default="INFO" )
    parser.add_option("-c", "--config", default=os.environ.get("QXML_CONFIG","no-qxml-config-envvar")  )
    parser.add_option("-k", "--key" ,   action="append" )
    parser.add_option("-v", "--val" ,   action="append" )
    parser.add_option("-i", "--inputfile"  )

    # used by transfer.py
    parser.add_option("-u", "--url"  )    
    parser.add_option("-t", "--target" , default='sys' )   

    cli, args = parser.parse_args()
    if not cli.inputfile and len(args)==1:
        cli.inputfile = args[0]	    

    if cli.key and cli.val:
        assert len(cli.key) == len(cli.val), "number of keys must match vals" 
	kv = dict(zip(cli.key,cli.val))
    else:
	kv = {}    
    return dict(config=cli.config,variables=kv,level=cli.level,inputfile=cli.inputfile,url=cli.url, target=cli.target), args 

def raw_parse_config(path):
    """
    ConfigParser does not support duplicate keys becoming lists
    so roll my own dinky parser

    Note the convention for lists of duplicating the duplicated key 
    in the section name::

	[dbxml]
	environment_dir = /tmp/dbxml
	default_collection = dbxml:////tmp/hfagc/hfagc.dbxml
	baseuri = 

	[container.path]
	path = /tmp/hfagc/hfagc.dbxml
	path = /tmp/hfagc/hfagc_system.dbxml

	[container.tag]
	tag = hfc
	tag = sys

	[namespace.name]
	name = rez 

	[namespace.uri]
	uri = http://hfag.phys.ntu.edu.tw/hfagc/rez


    This approach is used as it is straightforward to implement in C++ 
    with boost_program_options 

    """
    fp = open(path, "r" )
    cfg = {}
    sptn = re.compile("^\[(?P<sect>\S*)\]")
    vptn = re.compile("^(?P<var>\S*)\s*=\s*(?P<val>\S*)")
    sect = None
    for line in  fp.readlines():
	if line[0]=='#':continue     
        sm = sptn.match(line)
	if sm:
	    sect = sm.groupdict()['sect']	
	if sect:    
            vm = vptn.match(line)
	    if vm:
	        d = vm.groupdict()
	        var,val = d['var'],d['val']
                log.debug("sect %s var %s val %s " % ( sect, var, val ))
                key = "%s.%s" % (sect,var)
                if not sect.endswith(var):
		    cfg[key] = val	
                else:
                    if key not in cfg:
                        cfg[key] = [val]	
                    else:
                        cfg[key].append(val)	
		pass	
    return cfg


class Cfg(dict):
    def __repr__(self):return pformat(dict(self))

def kv_zip( cfg , kname, vname ):
    if kname in cfg and vname in cfg:
	keys,vals = cfg[kname], cfg[vname]
	assert len(keys) == len(vals)
        return dict(zip(keys,vals))
    else:
	return None    

def parse_config(path):
    log.info("reading config from %s " % path )
    raw = raw_parse_config(path)	
    cfg = Cfg()
    cfg['dbxml'] = dict((k,v) for k,v in raw.items() if k.startswith('dbxml.')) 
    cfg['containers'] = kv_zip(raw,"container.tag.tag","container.path.path") 
    cfg['namespaces'] = kv_zip(raw,"namespace.name.name","namespace.uri.uri") 
    cfg['source']     = kv_zip(raw,"container.tag.tag","container.source.source") 
    return cfg


def read_xquery( path ):
    placeholder = r"""
for $a in collection() return dbxml:metadata("dbxml:name", $a)
"""
    if not path:
	log.debug("using placeholder XQuery as no inputfile path specified")
	return placeholder
    lines = open(path, "r").readlines()
    if lines[0][0] == "#":  # comment 1st line when 1st char is '#' allowing shebang running  
        lines[0] = "(: %s :)\n" % lines[0].rstrip()
    q = "".join(lines)
    return q

def qxml_config():
    cli, args = parse_args()
    logging.basicConfig(level=getattr(logging, cli['level'].upper()))
    cfg = parse_config(cli['config'])
    variables = cli.pop('variables')
    cfg['cli'] = cli
    cfg['variables'] = variables
    log.debug(cfg)    
    cfg['query'] = read_xquery(cfg['cli']['inputfile'])
    return cfg



if __name__ == '__main__':
    cfg = qxml_config()
    print cfg 






