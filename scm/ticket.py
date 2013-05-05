#!/usr/bin/env python
"""
Querying trac.db concerning ticket changes
============================================

Hmm even the source sqlite has no group_concat::

	[blyth@cms01 e]$ sqlite3 /data/env/tmp/tracs/dybsvn/2013/04/25/104702/dybsvn/db/trac.db 
	[blyth@cms01 e]$ sqlite-
	[blyth@cms01 e]$ which sqlite3
	/data/env/system/sqlite/sqlite-3.3.16/bin/sqlite3
	[blyth@cms01 e]$ sqlite3 /data/env/tmp/tracs/dybsvn/2013/04/25/104702/dybsvn/db/trac.db 
	SQLite version 3.3.16
	Enter ".help" for instructions
	sqlite> select id, datetime(t.time,'unixepoch'), group_concat(c.author), summary from ticket t inner join ticket_change c on t.id = c.ticket group by c.ticket limit 10 ;
	SQL error: no such function: group_concat
	sqlite> 

	sqlite> 
	select c.ticket, t.summary, datetime(c.time, 'unixepoch'), c.author from ticket_change c, ticket t on c.ticket = t.id where c.author='blyth' order by c.time desc limit 100  ;




select 
   distinct sid as Account,
   (select value from session_attribute sa where name='name' and sa.sid=ss.sid ) AS Name,
   (select value from session_attribute sa where name='email' and sa.sid=ss.sid ) AS Email,
  datetime(last_visit,'unixepoch') AS Last_visit
from session ss where ss.authenticated=1
order by last_visit desc


"""
import logging
log = logging.getLogger(__name__)
from env.db.simtab import Table
from datetime import datetime

dt_ = lambda field, label:"datetime(%(field)s,'unixepoch') as %(label)s" % locals()
stamp2day_ = lambda stamp:datetime.fromtimestamp(float(stamp)).strftime("%Y-%m-%d")


class Ticket(object):
    tmpl = r"""
.. [ticket-%(id)s]
   %(summary)s
   %(lastchange)s  %(names)s
    """
    def __init__(self, id, summary, ttime):
        self.id = id
        self.summary = summary
        self.createtime = stamp2day_(ttime)        
        self.lastchange = stamp2day_(ttime)
        self._authors = []
        self._names = []
    def __repr__(self):
        self.authors = ",".join(self._authors)
        self.names = ", ".join(self._names)
        return self.tmpl % self.__dict__


def users_(tab):
    """
    :return: dict that associates short author sid with full names and emails
    """
    sql = """select 
                distinct sid as author,
               (select value from session_attribute sa where name='name' and sa.sid=ss.sid ) AS name,
               (select value from session_attribute sa where name='email' and sa.sid=ss.sid ) AS email
            from session ss where ss.authenticated=1 """
    users = {}
    for d in tab.listdict(sql, labels="author,name,email"):
         users[d['author']] = d
    return users

def tickets_(tab, tcut, tfmt, users):
    """
    Collects tickets and their contributors
    """
    tcut = datetime.strptime(tcut, tfmt).strftime("%s")
    cols = ["c.ticket as id","t.summary as summary","t.time as ttime", dt_("c.time", "_ctime"), "c.time as ctime" , "c.author as author" ]
    labels = "id,summary,ttime,_ctime,ctime,author" 

    from_ =  "from ticket_change c, ticket t on c.ticket = t.id"
    where_ = "where c.time > '%(tcut)s'" % locals()
    order_ = "order by c.time desc "

    sql = "select " + ",".join(cols) + " ".join(["",from_,where_,order_]) 
    log.info(sql)

    tkts = {}
    for d in tab.listdict( sql, labels=labels ):
        if d['id'] in tkts:
           tkt = tkts[d['id']]
        else: 
           tkt = Ticket(d['id'],d['summary'],d['ttime'])
           tkts[d['id']] = tkt
        pass
        tkt.lastchange = stamp2day_(d['ctime'])
        if d['author'] not in tkt._authors:
            tkt._authors.append(d['author'])
            tkt._names.append( users[d['author']]['name'] )

    log.info("tkt count %s " % len(tkts))
    return tkts

def main():
    dbpath = "/data/env/tmp/tracs/dybsvn/2013/04/25/104702/dybsvn/db/trac.db"
    author = 'blyth'
    tcut = "2012-08-01"
    tfmt = "%Y-%m-%d"

    logging.basicConfig(level=logging.INFO)

    tab = Table(dbpath) 
    users = users_(tab)
    tkts = tickets_(tab,tcut,tfmt, users)

    for id in sorted(tkts.keys()):
        tkt = tkts[id]
        if author in tkt._authors:
            print tkt

    
if __name__ == '__main__':
    main()
    




