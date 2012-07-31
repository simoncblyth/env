#!/usr/bin/env python
"""
At the end of writing a file, a CLOSE_WRITE event::

	CLOSE_WRITE event: /nas1/data/run00068.root (dest file)
	CLOSE_NOWRITE event: /nas1/data/TunnelData/V9/root/run00068.root (src file)

Event sequence of copying files from external HDD::

	CREATE event: /nas1/data/run02562.root
	OPEN event: /nas1/data/run02562.root
	MODIFY event: /nas1/data/run02562.root
	CLOSE_WRITE event: /nas1/data/run02562.root


"""
import pyinotify, logging
from analyze import analyze
log = logging.getLogger(__name__)


class MyEventHandler(pyinotify.ProcessEvent):
#    def process_IN_ACCESS(self, event):
#        print "ACCESS event:", event.pathname

    def process_IN_ATTRIB(self, event):
        print "ATTRIB event:", event.pathname

    def process_IN_CLOSE_NOWRITE(self, event):
        print "CLOSE_NOWRITE event:", event.pathname

    def process_IN_CLOSE_WRITE(self, event):
        print "CLOSE_WRITE event:", event.pathname
        analyze(event.pathname)

    def process_IN_CREATE(self, event):
        print "CREATE event:", event.pathname

    def process_IN_DELETE(self, event):
        print "DELETE event:", event.pathname

    def process_IN_MODIFY(self, event):
        print "MODIFY event:", event.pathname

    def process_IN_OPEN(self, event):
        print "OPEN event:", event.pathname

def main():
    # watch manager
    wm = pyinotify.WatchManager()

    # event handler
    eh = MyEventHandler()

    # notifier
    notifier = pyinotify.Notifier(wm, eh, read_freq=10)
    notifier.coalesce_events()
    wm.add_watch('/nas1/data/', pyinotify.ALL_EVENTS, rec=True)
    notifier.loop()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
