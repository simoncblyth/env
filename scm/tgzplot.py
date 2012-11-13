#!/usr/bin/env python
from __future__ import with_statement
import os, logging
log = logging.getLogger(__name__)
from datetime import datetime, timedelta

from env.plot.highstock import HSOptions


class TGZPlot(object):
    """
    
    """
    def __init__(self, tgz ):
        """
        :param tgz: `TGZ` instance
        """
        self.tgz = tgz    

    def hsopts(self, node ):
        """
        :param select: list of strings requires to be included in the names 
        """
        now = datetime.now().strftime("%c")
        hso = HSOptions()
        hso['series'] = []

        now = datetime.now()
        beg = now + timedelta(days=-30)
        hso['xmin'] = beg.strftime("%s000")
        hso['xmax'] = now.strftime("%s000")

        okdata = self.tgz.okdata( node )

        hok = dict(name="OK", 
                   data=okdata,
                 marker=dict(enabled=True, radius=3),
                tooltip=dict(valueDecimals=2), 
                  yAxis=1,
              )
        hso['series'].append( hok )

        for _ in self.tgz.items(node):
            name, dir = _
            data = self.tgz.data( node, _ )
            hss = dict(name=name, 
               data=data, 
             marker=dict(enabled=True, radius=3),
            tooltip=dict(valueDecimals=2), 
                ) 
            hso['series'].append( hss )
            log.info("append series %s %s of length %s  " % (name,dir, len(data)) )     
        pass
        pfx = self.tgz.pfx(node)     # do here as will already be cached following data access
        hso['renderTo'] = "container_%s" % node
        hso['title'] = "%s %s %s" % ( node, pfx, now )
        return hso


    def jsondump(self, path, node=None):
        """
        :param path: in which to dump the json series 
        """
        log.info("write json to %s " % path ) 
        hso = self.hsopts(node)
        with open(path,"w") as fp:
            fp.write(repr(hso))



if __name__ == '__main__':
    pass
    from env.scm.tgz import TGZ
    tgz = TGZ()
    plt = TGZPlot(tgz)
    node = 'C'
    hso = plt.hsopts(node)
    print hso 

