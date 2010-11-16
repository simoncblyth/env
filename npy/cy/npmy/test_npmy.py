"""
    * tinyints could be used for ladder/col/ring ... but leave at i4 initially   
    * voltage and pw go null quite a lot ... avoid for now by "not null" exclusion  

    * defaul mysql is not to '''use''' 
        store_result  : results stored in client
        use_result    : results stay on server, until fetched ..... for HUGE resultsets
        result = conn.store_result() old 1.2.2 API is not longer there in 1.3
         NB num_rows will not be valid for "use_result" until all rows are fetched

"""

import _mysql
import numpy as np

dt = np.dtype([('SEQNO', 'i4'), ('ROW_COUNTER', 'i4'), ('ladder', 'i4'),('col','i4'),('ring','i4'),('voltage','f4'),('pw','i4')])
sql = "select %(cols)s from %(tab)s limit %(limit)s " % { 'cols':",".join(dt.names), 'tab':"DcsPmtHv", 'limit':10 }

conn = _mysql.connect( read_default_file="~/.my.cnf", read_default_group="client" )
conn.query( sql )
result = conn.get_result()
n = result.num_rows()  

import npmy
#npmy.look(result)

a = np.zeros(10, dt)
npmy.fetch_rows_into_array_1( result, a )

print a


