#!/usr/bin/env python

import os

if __name__ == '__main__':

    print("<html>") 
    print("<ul>")
    for htm in filter(lambda p:p.endswith(".html"), os.listdir(".")):
        print("<li> <a href=\"%s\"> %s </a> </li>" % (htm, htm))
    pass
    print("</ul>")
    print("</html>") 


