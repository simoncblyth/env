#!/usr/bin/env python
"""
"""
import os
from rtfng.parser import RTFFile

if __name__ == '__main__':
    pass
    rf = RTFFile(os.path.abspath(os.path.expanduser("~/Desktop/msg.rtf")))
    print rf


