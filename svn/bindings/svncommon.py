#!/usr/bin/env python

import os, hashlib, logging
#import IPython as IP
log = logging.getLogger(__name__)

def mimic_svn_link_digest(link):
    assert os.path.islink(link)
    target = os.readlink(link)
    mimic = "link %s" % target 
    digest = hashlib.md5(mimic).hexdigest() 
    log.debug("link %s target %s mimic %s digest %s " % (link,target,mimic,digest)) 
    return digest 


def unprefix( paths, prefix, debug=False ):
    """ 
    :param paths: list of paths
    :param prefix: string prefix 
    :return: list of paths with prefix removed
    """
    is_not_blank_ = lambda _:len(_) > 0 
    is_prefixed_ = lambda _:_[0:len(prefix)] == prefix
    is_not_prefixed_ = lambda _:_[0:len(prefix)] != prefix
    unprefix_ = lambda _:_[len(prefix):] 

    non_blank = filter( is_not_blank_, paths )
    ppaths = filter( is_prefixed_ ,     non_blank )

    if debug:
        upaths = filter( is_not_prefixed_ , non_blank )
        print "upaths %s " % repr(upaths)
        print "ppaths %s " % repr(ppaths)

    #IP.embed() 
 
    assert len(ppaths) == len(non_blank), (len(ppaths),len(non_blank),"all non blank paths are expected to start with the prefix %s " % prefix )
    return map( unprefix_ , ppaths )   


