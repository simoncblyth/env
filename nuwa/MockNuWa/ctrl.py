#!/usr/bin/env python
"""

ctrl.py
=========

Creates the ctrl table, for mocknuwa scanning::

    sqlite> select * from ctrl ;
    id          max_blocks  max_steps   threads_per_block  seed        reset_rng_states
    ----------  ----------  ----------  -----------------  ----------  ----------------
    1           1024        30          32                 0           1               
    2           1024        30          64                 0           1               
    3           1024        30          96                 0           1               
    4           1024        30          128                0           1               
    5           1024        30          160                0           1               
    6           1024        30          192                0           1               
    7           1024        30          224                0           1               
    8           1024        30          256                0           1               
    9           1024        30          288                0           1               
    10          1024        30          320                0           1               
    11          1024        30          352                0           1               
    12          1024        30          384                0           1               
    13          1024        30          416                0           1               
    14          1024        30          448                0           1               
    15          1024        30          480                0           1               
    16          1024        30          512                0           1               
    sqlite> 



#. define envvar with `mocknuwa-;mocknuwa-export` 


"""
import logging, os, glob
log = logging.getLogger(__name__)
import numpy as np

class Table(dict):
    _drop = r"""
    drop table if exists %(table)s ;
    """
    _select = r"""
    select * from  %(table)s where id=%(id)s
    """
    drop = property(lambda self:self._drop % self )
    create = property(lambda self:self._create % self )
    insert = property(lambda self:self._insert % self )
    select = property(lambda self:self._select % self )

class Ctrl(Table):
    _create = r"""
    create table if not exists %(table)s (id integer primary key, max_blocks integer, max_steps integer, threads_per_block integer, seed integer, reset_rng_states integer );
    """ 
    _insert = r"""
    insert into %(table)s values (%(id)s, %(max_blocks)s, %(max_steps)s, %(threads_per_block)s, %(seed)s, %(reset_rng_states)s );
    """
    @classmethod
    def setup(cls, db):
        t = cls(table="ctrl")
        t['id'] = "null"
        t['max_blocks'] = "1024"
        t['max_steps'] = "30"
        t['threads_per_block'] = None  # filled below
        t['seed'] = "0"
        t['reset_rng_states'] = "1" 
         
        db.script(t.drop)
        db.script(t.create)
        for x in np.arange(32,512+1,32):
            t['threads_per_block'] = x 
            db.script(t.insert)


class Batch(Table):
    _create = r"""
    create table if not exists %(table)s (id integer primary key, tag text, path text);
    """
    _insert = r"""
    insert into %(table)s values (%(id)s, "%(tag)s", "%(path)s");
    """
    @classmethod
    def setup(cls, db):
        t = cls(table="tbatch")
        t['id'] = "null"

        db.script(t.drop)
        db.script(t.create)
        base = os.path.dirname(os.environ['DAE_PATH_TEMPLATE']) 
        for path in glob.glob(base + "/2014*.npy"):
            t['tag'] = os.path.basename(path)[:-4]
            t['path'] = path
            db.script(t.insert)
 
def main():
    logging.basicConfig(level=logging.INFO)
    from db import DB
    db = DB()
    Ctrl.setup(db)
    Batch.setup(db)


if __name__ == '__main__':
    main()







