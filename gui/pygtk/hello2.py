#!/usr/bin/env python
# example helloworld.py    http://wiki.tcl.tk/20238
import gtk
window = gtk.Window ()
window.connect('destroy', gtk.main_quit)
button = gtk.Button('Hello World')
button.connect('clicked', gtk.main_quit)
window.add(button)
window.show_all()
gtk.main()

