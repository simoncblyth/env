import os
from ConfigParser import SafeConfigParser as ConfigParser

class MyCnf(dict):
    def __init__(self, sect):
        cfp = ConfigParser()
        cfp.read(os.path.expanduser("~/.my.cnf"))
        self.update( dict( cfp.items(sect) ))

class MySQLNoDataDump(dict):
    _cmd = "mysqldump --no-defaults --no-data --host=%(host)s --user=%(user)s --password=%(password)s %(database)s > %(path)s "
    cmd = property( lambda self:self._cmd % self )
    _path = "~/%(database)s.nodata.sql"
    path = property( lambda self:self._path % self )
    def __call__(self):
       self.update(path=self.path) 
       print self.cmd
       print os.popen(self.cmd).read() 

if __name__=='__main__':
    ndd = MySQLNoDataDump(MyCnf("dcs"))
    ndd() 

