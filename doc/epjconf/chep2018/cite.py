#!/usr/bin/env python
"""
Hmm bibtex turns out to be useless, it messes with the authors
in funny ways...  Thought about generating the 

epsilon:chep2018 blyth$ cat opticks-blyth-chep2018-v1.tex | ./cite.py 
opticksURL
opticksGroup
g4A
g4B
g4C
optixPaper
...
"""

import sys, re, logging, argparse

class Tex(list):
    CITE_PTN = re.compile("\cite{(\S*?)}")
    def __init__(self, lines):
        self.lines = lines
        self.cites = []
        for line in lines:
            if line[0] == '%':continue
            if line.find("\cite") != -1:
                cites = self.CITE_PTN.findall(line)
                self.cites.extend(cites)
            pass
        pass

    def __str__(self):
        return "".join(self.lines)
    def __repr__(self):
        return "\n".join(self.cites)

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--bib",  help="Bib file" )
    args = parser.parse_args()

    lines = sys.stdin.readlines()
    tex = Tex(lines)
    print(repr(tex))

