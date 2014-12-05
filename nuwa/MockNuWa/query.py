#!/usr/bin/env python
"""

sqlite> select batch_id, nwork, min(ctrl_id), max(ctrl_id) from log group by batch_id having nwork > 1000 ;
batch_id    nwork       min(ctrl_id)  max(ctrl_id)
----------  ----------  ------------  ------------
3           1888        1             11          
12          3095        1             11          
13          2053        1             11          
14          1869        1             11          
19          4585        1             11          
24          2779        1             11          
25          2025        1             11          
34          3159        1             11          
38          2553        1             10          
41          3201        1             10          
44          2979        1             10          
45          2463        1             10          
sqlite> 

big = q("select distinct(batch_id) from log where nwork>1000")
ibig = map(int,big[:,0])


def bidplot(bid):
    a = q("select id, tottime from log where batch_id=%s" % bid )
    plt.scatter(a[:,0],a[:,1])
    plt.show()


In [72]: map(bidplot, ibig)






In [45]: import pylab

In [47]: pylab.ion()   ## avoids plt.show() blocking, and doesnt clear the plot to collect entries 

In [58]: a = q("select id, tottime from log where batch_id=19") ; plt.scatter(a[:,0],a[:,1]) ; plt.show()
npar: envvar SQLITE3_DATABASE:/usr/local/env/nuwa/mocknuwa.db ncol 2 nrow 14 type f  fbufmax 1000  



sqlite> select log.id, ctrl_id, batch_id, nwork, threads_per_block, tottime from log, ctrl on log.ctrl_id = ctrl.id  where batch_id = 11 ;
id          ctrl_id     batch_id    nwork       threads_per_block  tottime   
----------  ----------  ----------  ----------  -----------------  ----------
31          1           11          5           32                 0.0066    
79          2           11          5           64                 0.004402  
127         3           11          5           96                 0.004013  
175         4           11          5           128                0.003268  
191         1           11          5           32                 0.006594  
239         2           11          5           64                 0.004413  
287         3           11          5           96                 0.00351   
335         4           11          5           128                0.003515  
383         5           11          5           160                0.002908  
431         6           11          5           192                0.002829  
479         7           11          5           224                0.002833  
527         8           11          5           256                0.002829  
575         9           11          5           288                0.002813  
sqlite> 


sqlite> select log.id, ctrl_id, batch_id, nwork, threads_per_block, tottime from log, ctrl on log.ctrl_id = ctrl.id  where batch_id = 12 ;
id          ctrl_id     batch_id    nwork       threads_per_block  tottime   
----------  ----------  ----------  ----------  -----------------  ----------
32          1           12          3095        32                 0.586859  
80          2           12          3095        64                 0.389117  
128         3           12          3095        96                 0.321172  
192         1           12          3095        32                 0.604534  
240         2           12          3095        64                 0.380468  
288         3           12          3095        96                 0.331132  
336         4           12          3095        128                0.3213    
384         5           12          3095        160                0.266781  
432         6           12          3095        192                0.269787  
480         7           12          3095        224                0.268663  
528         8           12          3095        256                0.270102  
576         9           12          3095        288                0.264849  
624         10          12          3095        320                0.27768   
sqlite> 








sqlite> select nwork, tottime, threads_per_block, tag from log, ctrl, batch on log.ctrl_id = ctrl.id and log.batch_id = batch.id ;


sqlite> select log.id, locdt, nwork, tottime, threads_per_block, tag from log, ctrl, batch on log.ctrl_id = ctrl.id and log.batch_id = batch.id where batch_id = 1 ;
id          locdt                nwork       tottime     threads_per_block  tag            
----------  -------------------  ----------  ----------  -----------------  ---------------
1           2014-12-05 15:32:54  445         0.069575    32                 20140514-174932
2           2014-12-05 16:42:45  445         0.070106    32                 20140514-174932
3           2014-12-05 16:58:18  445         0.069926    32                 20140514-174932
4           2014-12-05 17:09:00  445         0.070019    32                 20140514-174932
5           2014-12-05 18:38:14  445         0.105667    32                 20140514-174932
6           2014-12-05 18:38:15  445         0.075418    64                 20140514-174932
7           2014-12-05 18:38:16  445         0.070672    96                 20140514-174932
8           2014-12-05 18:38:18  445         0.06384     128                20140514-174932
9           2014-12-05 18:38:20  445         0.056004    160                20140514-174932
10          2014-12-05 18:38:21  445         0.052133    192                20140514-174932
11          2014-12-05 18:38:23  445         0.047045    224                20140514-174932
12          2014-12-05 18:38:25  445         0.045302    256                20140514-174932
13          2014-12-05 18:38:27  445         0.04511     288                20140514-174932
14          2014-12-05 18:38:31  445         0.045213    320                20140514-174932
15          2014-12-05 18:38:35  445         0.045314    352                20140514-174932
16          2014-12-05 18:38:40  445         0.045938    384                20140514-174932
17          2014-12-05 18:38:45  445         0.045929    416                20140514-174932
18          2014-12-05 18:38:52  445         0.045956    448                20140514-174932
19          2014-12-05 18:38:59  445         0.045953    480                20140514-174932
20          2014-12-05 18:39:07  445         0.046014    512                20140514-174932
21          2014-12-05 19:10:12  445         0.105254    32                 20140514-174932
69          2014-12-05 19:11:14  445         0.069973    64                 20140514-174932
117         2014-12-05 19:12:18  445         0.057519    96                 20140514-174932
165         2014-12-05 19:13:24  445         0.056774    128                20140514-174932
sqlite> 

sqlite> select log.id, tottime, threads_per_block from log, ctrl on ctrl_id = ctrl.id  where batch_id = 1 ;
id          tottime     threads_per_block
----------  ----------  -----------------
1           0.069575    32               
2           0.070106    32               
3           0.069926    32               
4           0.070019    32               
5           0.105667    32               
6           0.075418    64               
7           0.070672    96               
8           0.06384     128              
9           0.056004    160              
10          0.052133    192              
11          0.047045    224              
12          0.045302    256              
13          0.04511     288              
14          0.045213    320              
15          0.045314    352              
16          0.045938    384              
17          0.045929    416              
18          0.045956    448              
19          0.045953    480              
20          0.046014    512              
21          0.105254    32               
69          0.069973    64               
117         0.057519    96               
165         0.056774    128              
sqlite> 




"""
