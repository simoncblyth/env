
# from chroma.rootimport

# TApplication is so incredibly annoying.
# STOP SNOOPING THE ARGUMENT LIST AND PRINTING BOGUS HELP MESSAGES!! 
import sys 
_argv = sys.argv
sys.argv = []

try:
    import ROOT
    # Have to do something with the module to get TApplication initialized
    ROOT.TObject # This just fetches the class and does nothing else
except ImportError:
    ROOT = None


sys.argv = _argv
del _argv


