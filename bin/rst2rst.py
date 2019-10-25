#!/usr/bin/env python

from __future__ import print_function
import os, sys


class R2R(object):
    title_mkr = "-" * 10 , "=" * 10 
    talk_mkr = ".. s5_talk::"

    @classmethod
    def make_title(cls, title, extra=""):
        tx = "%s %s" % (title, extra) 
        return [tx, "-" * len(tx) ]

    @classmethod
    def make_talk_slide(cls, title, size="normal"):
        tt = cls.make_title(title, extra="Talk") 
        tt.append("")
        tt.append(".. class:: %s" % size)
        return tt 

    def __init__(self, path):
        lines = map(lambda _:_[:-1], file(path).readlines())
        self.titles = []
        self.curtitle = None
        self.olines = self.parse(lines)

    def parse(self, lines):
        olines = [] 
        for i in range(len(lines)-2):
            if lines[i+1].startswith(self.title_mkr[0]) or lines[i+1].startswith(self.title_mkr[1]):
                title = lines[i]
                self.titles.append(title)
                self.curtitle = title
                print(title)
            pass
            if lines[i].startswith(self.talk_mkr):

                size = "normal"
                if lines[i+2].find("SMALL") > -1:
                    size = "small"
                pass
                ts = self.make_talk_slide(self.curtitle, size=size)
                olines.extend(ts)
            else:
                olines.append(lines[i])
            pass
        pass
        olines.append(lines[-2])
        olines.append(lines[-1])
        return olines 
    def __str__(self):
        return "\n".join(self.olines)


if __name__ == '__main__':
     path = sys.argv[1]

     name = os.path.basename(path)
     stem, ext = os.path.splitext(name)
     assert not stem.endswith("_TALK"), (path, name, stem, ext)  

     assert ext == ".txt" 
     oname = "%s_TALK.txt" % stem

     r2r = R2R(path)
     out = file(oname, "w")
     print("writing to %s " % oname )
     print(r2r, file=out)

     





