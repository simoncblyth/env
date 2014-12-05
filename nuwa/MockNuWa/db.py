#!/usr/bin/env python
"""
"""

import os, logging
log = logging.getLogger(__name__)
import sqlite3
from collections import OrderedDict as odict


class DB(object):
    def __init__(self, path=os.environ['SQLITE3_DATABASE']):
        log.info("connecting to %s" % path)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        self.c = c

    def not_working_insert_statement_0(self, table, d):
        return "".join(["insert into %s " % table ,
                        "(",
                            ",".join(d.keys()),
                        ") ",
                        "values(",
                           ",".join(map(lambda k:":%s"%k,d.keys())),
                        ")"])

    def not_working_insert_statement_1(self, table, d):

        def value_(v):
            return "null" if v is None else str(v) 

        return "".join(["insert into %s " % table ,
                        "(",
                            ",".join(d.keys()),
                        ") ",
                        "values(",
                           ",".join(map(value_,d.values())),
                        ")"])



    def insert(self, table, d ):
        sql = self.insert_statement_1(table, d)
        self.c.executescript(sql)

    def script(self, sql):
        self.c.executescript(sql)

    def __call__(self, sql):
        self.c.execute(sql)
        return self.c.fetchall()


if __name__ == '__main__':
    pass


