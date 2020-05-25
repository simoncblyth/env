#!/usr/bin/env python

from __future__ import print_function
import os, sys, logging
log = logging.getLogger(__name__)


class R2R(object):
    title_mkr = "-" * 10 , "=" * 10 
    talk_mkr = ".. s5_talk::"

    @classmethod
    def make_title(cls, title, extra=""):
        """
        :param title:
        :param extra: text to append to the title
        :return list: of length 2 with title and extra and "----" title marker   
        """
        tx = "%s %s" % (title, extra) 
        return [tx, "-" * len(tx) ]

    @classmethod
    def make_talk_slide(cls, title, size="normal"):
        """
        :param title:
        :param size:
        :return tt: 
        """
        tt = cls.make_title(title, extra="Talk") 
        tt.append("")
        tt.append(".. class:: %s" % size)
        return tt 

    def __init__(self, path):
        """
        :param path: to RST .txt file 
        """
        lines = map(lambda _:_[:-1], file(path).readlines())
        self.titles = []
        self.talktitles = []
        self.curtitle = None
        self.olines = self.parse(lines)
        self.check(self.titles, self.talktitles)

    def check(self, a, b ):
        ab = set(a) - set(b)
        ba = set(b) - set(a)
        assert len(ba) == 0, ba
        log.info("ab : %d slide pages without s5_talk " % len(ab) )
        for i, t in enumerate(a):
            st = " " if t in b else "X" 
            print("%2d : %1s : %s" % (i, st, t ))
        pass

    def parse(self, lines):
        """
        :param lines: content of RST .txt source file

        * forms output olines with ".. s5_talk::" replaced with new slide page titles
          based on the current title with Talk appended

        * line pairs scan to collect titles by looking for 
          the "----------" or "==========" on the line after 

        * when line starts with ".. s5_talk::" do not append
          by adding a talk slide

        """
        olines = [] 
        log.debug("line pair scan for titles")
        for i in range(len(lines)-2):
            if lines[i+1].startswith(self.title_mkr[0]) or lines[i+1].startswith(self.title_mkr[1]):
                title = lines[i]
                self.titles.append(title)
                self.curtitle = title
                #print(title)
            pass
            if lines[i].startswith(self.talk_mkr):
                self.talktitles.append(self.curtitle)
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
     logging.basicConfig(level=logging.INFO)

     log.info(sys.argv[0])

     name = os.path.basename(path)
     stem, ext = os.path.splitext(name)
     assert not stem.endswith("_TALK"), (path, name, stem, ext)  

     assert ext == ".txt" 
     oname = "%s_TALK.txt" % stem

     r2r = R2R(path)
     out = file(oname, "w")
     print("writing to %s " % oname )
     print(r2r, file=out)

     





