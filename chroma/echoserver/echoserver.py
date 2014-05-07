#!/usr/bin/env python
"""
Invoke with `echoserver.sh` for envvar setup.
"""
import os, time, logging, argparse
log = logging.getLogger(__name__)

from env.chroma.ChromaPhotonList.responder import CPLResponder

def parse(doc):

    defaults = {
                'bind':os.environ['ECHO_SERVER_CONFIG'],
                'timeout':100,
                'sleep':0.5,
                'random':False,
                'dump':False,
              } 

    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-l","--level", default="INFO")
    parser.add_argument(     "--format", default="%(asctime)-15s %(name)-20s:%(lineno)-3d %(message)s")

    parser.add_argument(     "--bind",    help="Network endpoint to bind to. Default %(default)s" )
    parser.add_argument(     "--timeout", type=int, help="Timeout to wait for messages (integer milliseconds), when <=0 waits forever. Default %(default)s " )
    parser.add_argument(     "--sleep",   type=float, help="Time to sleep (float seconds) after receiving messages. Default %(default)s " )
    parser.add_argument(     "--random",  action="store_true", help="Reply with random ChromaPhotonList rather than echoing. Default %(default)s " )
    parser.add_argument(     "--dump",    action="store_true", help="Dump the ChromaPhotonList received. Default %(default)s " )

    parser.set_defaults( **defaults ) 
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.level.upper()), format=args.format)
    args.timeout = None if args.timeout <= 0 else args.timeout 
    return args 


def main():
    config = parse(__doc__)
    responder = CPLResponder( config )
    while True:
        responder.poll()



if __name__ == '__main__':
    main()
   




