#!/usr/bin/env python

import os
import sys
dir = os.path.abspath(os.path.dirname(sys.argv[0])) 

import django
from django.template.loader import render_to_string
django.conf.settings.configure(TEMPLATE_DIRS=[dir])
from django.template import Context

if __name__=='__main__':
    import sys 
    from parse import Tab
    t = Tab.parse_csv( sys.stdin )
    c = Context( dict(t=t) )
    ## add shortcut dict with frequently used qtys to the context
    print render_to_string('SubDbiTableRow.h', { 'cls': t.meta.get('class','ErrorNoClass') } , c )






