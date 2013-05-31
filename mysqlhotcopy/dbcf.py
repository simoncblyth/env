#!/usr/bin/env python
"""


A per-SEQNO digest that can be compared between tables in separate DBs that should be the same::

    mysql> select SEQNO,md5(group_concat(md5(concat_ws(",",ROW_COUNTER,RUNNO,FILENO,CHANNELID,OCCUPANCY,DADCMEAN,DADCRMS,HVMEAN,HVRMS)) separator ",")) as digest from DqChannel group by SEQNO limit 10 ;
    +-------+----------------------------------+
    | SEQNO | digest                           |
    +-------+----------------------------------+
    |     1 | 234d05bac921e752a830b725b8e7025d | 
    |     2 | 7c47447ac58bf99cfb1e1619d1ae497b | 
    |     3 | 74dd38ac411bb10f61288f871d9c9bf1 | 
    |     4 | 0ed802219ade8d9aa3b3033d75b2f62f | 
    |     5 | 32a76390c03c6a4bd3d0d1a958725238 | 
    |     6 | 06a6d73226d390e1556450edd8fd54ec | 
    |     7 | ebcdbfb042bf60e934de2d5ee0ec84db | 
    |     8 | d96107391a788a147e861e9c154d9258 | 
    |     9 | 57dac2ede0fe9e48be87896480fd6d84 | 
    |    10 | 29e1fec1affc0ebf814af1777c455078 | 
    +-------+----------------------------------+
    10 rows in set (0.03 sec)




"""
import os, logging
from pprint import pformat
log = logging.getLogger(__name__)
from db import DB



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)-8s %(message)s" )
    db = DB("client", group_by="SEQNO")

    a = "tmp_ligs_offline_db_0.DqChannel"
    b = "channelquality_db.DqChannel"

    db.digest_table_scan( a, b )



