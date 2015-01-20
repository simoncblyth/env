#!/usr/bin/env python
"""
DAEConfigBase
================

"""
import os, sys, logging, math, argparse
import numpy as np

try: 
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

log = logging.getLogger(__name__)


class ArgumentParserError(Exception): pass
class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)

class ConfigBase(object):
    """
    Classes specializing this expected to implement
    the below two methods which return the parser and defaults.

    * _make_base_parser
    * _make_live_parser

    At instantiation the two parsers are hooked up and defaults combined

    """
    def __init__(self, doc):
        """
        :param doc:
        """
        base_parser, base_defaults = self._make_base_parser(doc)
        init_parser, live_defaults = self._make_live_parser(parents=[base_parser]) 

        self.init_parser = init_parser

        defaults = OrderedDict()
        defaults.update(base_defaults)
        defaults.update(live_defaults)

        self.base_defaults = base_defaults
        self.live_defaults = live_defaults
        self.defaults = defaults

        live_parser, dummy         = self._make_live_parser(argument_default=argparse.SUPPRESS, parents=[], with_defaults=False) 
        self.live_parser = live_parser
        self.args = None

    def init_parse(self):
        try:
            args = self.init_parser.parse_args()
        except ArgumentParserError, e:
            print "FATAL : ArgumentParserError %s %s " % (e, repr(sys.argv)) 
            self.init_parser.print_help()
            print "FATAL : ArgumentParserError %s %s " % (e, repr(sys.argv)) 
            sys.exit(1)
        
        logging.basicConfig(level=getattr(logging, args.loglevel.upper()), format=args.logformat )
        np.set_printoptions(precision=4, suppress=True)
        self.args = args

    def live_parse(self, cmdline):
        live_args = None           
        try:
            live_args = self.live_parser.parse_args(cmdline.lstrip().rstrip().split(" "))
        except ArgumentParserError, e:
            log.info("ArgumentParserError %s while parsing %s " % (e, cmdline)) 
        pass
        return live_args

    def __call__(self, cmdline):
        return self.live_parse(cmdline)

    def _settings(self, args, defaults, all=False):
        if args is None:return "PARSE ERROR"
        if all:
            filter_ = lambda kv:True
        else:
            filter_ = lambda kv:kv[1] != getattr(args,kv[0]) 
        pass
        wid = 20
        fmt = " %-30s : %20s : %s %20s %s "
        mkr_ = lambda k:"**" if getattr(args,k) != defaults.get(k) else "  "
        return "\n".join([ fmt % (k,str(v)[:wid],mkr_(k),str(getattr(args,k))[:wid],mkr_(k)) for k,v in filter(filter_,defaults.items()) ])

    def base_settings(self, all_=False):
        return self._settings( self.args, self.base_defaults, all_ )

    def live_settings(self, all_=False):
        return self._settings( self.args, self.live_defaults, all_ )


    def report(self):
        changed = self.changed_settings()
        if len(changed.split("\n")) > 1:
            print "changed settings\n", changed
        #print "all settings\n",self.all_settings()

    def all_settings(self):
        return "\n".join(filter(None,[
                      self.base_settings(True) ,
                      "---", 
                      self.live_settings(True) 
                         ]))
    def changed_settings(self):
        return "\n".join(filter(None,[
                      self.base_settings(False) ,
                      "---", 
                      self.live_settings(False) 
                         ]))

    def __repr__(self):
        return self.changed_settings() 
 
