#!/usr/bin/env python
import sys

from optparse import OptionParser


def main(args):

    parser = OptionParser()
    choices=["test","summarize"]
    parser.add_option("--action"  ,   default="test" , choices=choices , help="choose one of: %s   default:[%%default] " % ", ".join(choices) )
    parser.add_option("--html"  ,   action="store_true"   ,   help="generate test results in html, xml creation is forced when this option is chosen, default:[%default]" )
    parser.set_defaults( html=False )


if __name__=='__main__':
    sys.exit(main(sys.argv))

