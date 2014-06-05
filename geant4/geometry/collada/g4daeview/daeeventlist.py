#!/usr/bin/env python
"""
TODO: test in case of live additions/deletions
"""
import logging
from glob import glob
log = logging.getLogger(__name__)


from daemenu import DAEMenu


class DAEEventListMenu(DAEMenu):
    def __init__(self, handler, menu_callback ):
        DAEMenu.__init__(self, "eventlist")
        curpath = handler.path
        for path in handler.paths:
            title = "*%s*" % path if path == curpath else path
            self.add( title, menu_callback, path=path )
        pass



class DAEEventList(object):
    """
    A list of paths and a "cursor" index pointing at one of them
    """
    def __init__(self, path_template):
        self.path_template = path_template
        self._index = 0
        self.paths = self.find_paths()

    def update(self):
        """
        Need to hold the curpath whilst update as
        the index may well change, or become None if path was deleted 
        or template changed.

        The path getter and setter take care of changing the index
        """
        curpath = self.path
        self.paths = self.find_paths()
        self.path = curpath

    def resolve(self, arg):
        return self.path_template % {'arg':arg } 

    def find_paths(self):
        return glob(self.resolve("*"))

    def abspath(self, path_):
        if path_[0] == '/':
            path = path_
        else: 
            path = self.resolve(path_)
        return path

    def path_at_index(self, index):
        try:
            path = self.paths[index]
        except IndexError:
            path = None
        return path 

    def _set_path(self,path):
        if path is None: 
           return
        _index = self.find(path)
        if _index is None:
           log.warn("didnt find path %s : try updating first " % path ) 
           return
        self._index = _index
    def _get_path(self):
        return self.path_at_index(self._index)
    path = property(_get_path, _set_path) 

 
    def find(self, path_):
        """
        :return: index of path in the list or None if not found
        """
        path = self.abspath(path_) 
        try:
            index = self.paths.index(path)
        except ValueError:
            index = None
        return index

    def _get_prev(self):
        """
        Using this getter changes the cursor
        """
        _index = (self._index - 1)
        if _index < 0:
            return None
        _path = self.path_at_index(_index)
        self.path = _path
        return _path
    prev = property(_get_prev)

    def _get_next(self):
        """
        Using this getter changes the cursor
        """
        _index = (self._index + 1) 
        if _index > len(self.paths):
            return None
        _path = self.path_at_index(_index)
        self.path = _path
        return _path
    next_ = property(_get_next)


if __name__ == '__main__':
    import os
    logging.basicConfig(level=logging.INFO)
    evl = DAEEventList( os.environ['DAE_PATH_TEMPLATE'] ) 
    print evl 

    evl.path = "1"
    evl.path = "20140514-180119"

    print "next_"
    while evl.next_:
        print evl.path
        
    print "prev"
    while evl.prev:
        print evl.path
 



