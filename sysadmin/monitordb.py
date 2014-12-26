#/usr/bin/env python
"""
Commandline usage::

   monitordb.sh percent_free --limit 240 --level debug

   monitordb.sh gb_free --limit 72 --level debug


IPython Usage::

   SQLITE3_DATABASE=~/.env/C_envmon.sqlite ipython.sh 

   from env.sysadmin.monitordb import MonitorDB

   db = MonitorDB()
   a = db.history("gb_free", 1000)

   plt.scatter(a[:,0],a[:,1])
   plt.show()


"""
import os, sqlite3, datetime, logging, time
import numpy as np

log = logging.getLogger(__name__)
os.environ.setdefault('SQLITE3_DATABASE',"~/.env/C_envmon.sqlite")

class MonitorDB(object):
    timefmt="%Y-%m-%dT%H:%M:%S"
    def __init__(self, config):
        self.config = config
        path = os.path.expanduser(config.path)
        log.info("path %s " % path)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        self.c = conn.cursor()
        
    def __call__(self, sql):
        self.c.execute(sql)
        return self.c.fetchall()

    def history(self, key="gb_free", limit=1000 ):
        sql = "select date, ret from %s order by date desc limit %s ;" % (self.config.table, limit) 
        log.info(sql)
        rows = self(sql) 
        tvs = []
        for row in rows:
            ts = time.strptime(row[0],self.timefmt)
            t = time.mktime(ts)
            dt = datetime.datetime.fromtimestamp(t)
            sd = row[1]
            assert sd[0] == '{' and sd[-1] == '}'

            log.debug(sd)
            d = eval(sd)
            v = float(d[key])
            tv = [t, v]
            tvs.append(tv)
        pass
        return np.array(tvs)



def parse_args(doc):
    import argparse
    parser = argparse.ArgumentParser(doc)

    d = {}
    d['level'] = "INFO"
    d['ipython'] = False

    d['var'] = "SQLITE3_DATABASE"
    d['limit'] = 1000
    d['table'] = "diskmon" 

    parser.add_argument("-l","--level", default=d['level'], help="INFO/DEBUG/WARN/..")  
    parser.add_argument("--ipython", action="store_true", default=d['ipython'] ) 

    parser.add_argument("--var", default=d['var'], help="broker envvar pointing to db")  
    parser.add_argument("--table", default=d['table'] ) 
    parser.add_argument("--limit", default=d['limit'] ) 
    parser.add_argument("keys", nargs="+", default=["gb_free"] ) 
    
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.level.upper()))
    np.set_printoptions(precision=3, suppress=True)
    return args



def main():
    import matplotlib.pyplot as plt

    config = parse_args(__doc__)
    config.path = os.environ[config.var]

    db = MonitorDB(config)

    for key in config.keys:
        a = db.history(key, config.limit)
        plt.scatter(a[:,0],a[:,1])
        plt.show()  



if __name__ == '__main__':
    main()





