#!/usr/bin/env python

import os
import sys
dir = os.path.abspath(os.path.dirname(sys.argv[0]) + os.sep +'templates') 

def filltmpl(tmpl, ctx, **kwa ):
	"""
	       tmpl   :  name of template file, that will be loaded from the same dir as this module
	        ctx   :  context parsed from the spec file
	        kwa   :  extra context, eg shortcuts to frequently used qtys 
	""" 
	import django
	from django.template import Context
	from django.template.loader import render_to_string
	django.conf.settings.configure(TEMPLATE_DIRS=[dir])
	return render_to_string( tmpl, kwa, Context({'t':ctx}) )
	

if __name__=='__main__':
    import sys 
    from parse import Tab
    t = Tab.parse_csv( sys.stdin )
    print filltmpl( 'SubDbiTableRow.h', t , cls=t.meta.get('class','ErrorNoClass'))

    
    






