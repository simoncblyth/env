
#!/usr/bin/env python

import logging
import os
import sys

from nose.plugins.base import Plugin
#from nose.plugins.plugintest import run_buffered as run
from nose.core import run

from optparse import OptionGroup


logging.basicConfig(level=logging.DEBUG)
log =  logging.getLogger(__name__)



class DybWrap(Plugin):
    """ placing Dyb customizations into a plugin ??? 
    
       try to adopt the pattern of 
           http://www.somethingaboutorange.com/mrl/projects/nose/doc/plugin_prof.html
           http://www.somethingaboutorange.com/mrl/projects/nose/doc/plugin_cover.html
    
       actually can do it all in the one plugin...
          perhaps
          
          DybWrap(XmlOutput)
             probably need to build up a DOM of the report rather than blindly dumping  as at present 
             and then output this to the requested place ... file or stdout it in the report 
    
    
    """
    def options(self, parser, env=os.environ):
        Plugin.options(self, parser, env)
                
        group = OptionGroup(parser, "Dyb Customization ",  "Caution: use these options at your own risk.  "  )

        
        choices=["test","summarize"]

        group.add_option("--action"  ,   default="test" , choices=choices , help="choose one of: %s   default:[%%default] " % ", ".join(choices) )

        group.add_option("--tofile"    , default=False                           , help="write outputs to file rather than stdout, default:[%default] " ) 
        group.add_option("--reportdir" , default="tests/nose"     , type="string" , help="absolute or relative to searchdir path to store output, default:[%default]  ")  

        group.add_option("--dryrun",     action="store_true" ,  help="discover and report tests found but do not run them, default:[%default] ")    
        parser.set_defaults( dryrun=False ) 

        group.add_option("--xml"   ,   action="store_true"  ,    help="generate test results in xml, default:[%default] "  )
        parser.set_defaults( xml=False )
    
        group.add_option("--html"  ,   action="store_true"   ,   help="generate test results in html, xml creation is forced when this option is chosen, default:[%default]" )
        parser.set_defaults( html=False )
    
        group.add_option("--auto"  ,  action="store_true"   ,     help="collective option that implies --tofile, --html, --xml, --quiet , default:[%default]")
        parser.set_defaults( auto=False )


        parser.add_option_group(group)

        # would be nicer to avoid this script having to knowing too much about Dyb layouts / CMT etc.. 
        #  get it to do the right thing via invokation from the right working directory  
        #op.add_option("--siteroot" , default=os.environ['SITEROOT'] )

    def configure(self, options, config):
        log.debug("Dyb configure")
        Plugin.configure(self, options, config)
        self.config = config

    def begin(self):
        """ begin   """
        print "Dyb begin "
        log.debug("Dyb begin")

    def report(self, stream):
        print "Dyb report "
        tmp = sys.stdout
        sys.stdout = stream
        
        try:
            print "THE REPORT "
        finally:
            sys.stdout = tmp
        
        log.debug("Dyb report")



if '__main__'==__name__:
    run(argv=sys.argv,  plugins=[DybWrap()])






