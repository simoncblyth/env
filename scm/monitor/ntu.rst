cfg: {'HUB': 'C2R',
 'dbpath': '$LOCAL_BASE/env/scm/scm_backup_monitor.db',
 'email': 'blyth@hep1.phys.ntu.edu.tw',
 'jspath': '$APACHE_HTDOCS/data/scm_backup_monitor_%(node)s.json',
 'select': 'repos/env tracs/env repos/aberdeen tracs/aberdeen repos/tracdev tracs/tracdev repos/heprez tracs/heprez',
 'srvnode': 'cms02'}
monitor cfg: {'HOST': 'C',
 'HUB': 'C2R',
 'dbpath': '$LOCAL_BASE/env/scm/scm_backup_monitor.db',
 'email': 'blyth@hep1.phys.ntu.edu.tw',
 'jspath': '$APACHE_HTDOCS/data/scm_backup_monitor_%(node)s.json',
 'select': 'repos/env tracs/env repos/aberdeen tracs/aberdeen repos/tracdev tracs/tracdev repos/heprez tracs/heprez',
 'srvnode': 'cms02'} 
[C] run: find $SCM_FOLD/backup/cms02 -name '*.gz' -exec du --block-size=1M {} \;
[C] out: 2	/data/var/scm/backup/cms02/tracs/aberdeen/2012/08/02/171121/aberdeen.tar.gz
[C] out: 2	/data/var/scm/backup/cms02/tracs/aberdeen/2012/08/01/171121/aberdeen.tar.gz
[C] out: 2	/data/var/scm/backup/cms02/tracs/aberdeen/2012/08/03/171121/aberdeen.tar.gz
[C] out: 2	/data/var/scm/backup/cms02/tracs/aberdeen/2012/07/31/171120/aberdeen.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/tracs/data/2012/04/30/130103/data.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/tracs/data/2012/04/29/130108/data.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/tracs/data/2012/04/28/130107/data.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/tracs/data/2012/05/01/130104/data.tar.gz
[C] out: 9	/data/var/scm/backup/cms02/tracs/heprez/2012/08/02/171121/heprez.tar.gz
[C] out: 9	/data/var/scm/backup/cms02/tracs/heprez/2012/08/01/171121/heprez.tar.gz
[C] out: 9	/data/var/scm/backup/cms02/tracs/heprez/2012/08/03/171121/heprez.tar.gz
[C] out: 9	/data/var/scm/backup/cms02/tracs/heprez/2012/07/31/171120/heprez.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/tracs/newtest/2012/08/02/171121/newtest.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/tracs/newtest/2012/08/01/171121/newtest.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/tracs/newtest/2012/08/03/171121/newtest.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/tracs/newtest/2012/07/31/171120/newtest.tar.gz
[C] out: 52	/data/var/scm/backup/cms02/tracs/env/2012/08/02/171121/env.tar.gz
[C] out: 52	/data/var/scm/backup/cms02/tracs/env/2012/08/01/171121/env.tar.gz
[C] out: 52	/data/var/scm/backup/cms02/tracs/env/2012/08/03/171121/env.tar.gz
[C] out: 52	/data/var/scm/backup/cms02/tracs/env/2012/07/31/171120/env.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/tracs/tracdev/2012/08/02/171121/tracdev.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/tracs/tracdev/2012/08/01/171121/tracdev.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/tracs/tracdev/2012/08/03/171121/tracdev.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/tracs/tracdev/2012/07/31/171120/tracdev.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/folders/svnsetup/2012/08/02/171121/svnsetup.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/folders/svnsetup/2012/08/01/171121/svnsetup.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/folders/svnsetup/2012/08/03/171121/svnsetup.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/folders/svnsetup/2012/07/31/171120/svnsetup.tar.gz
[C] out: 168	/data/var/scm/backup/cms02/repos/aberdeen/2012/08/02/171121/aberdeen-1715.tar.gz
[C] out: 168	/data/var/scm/backup/cms02/repos/aberdeen/2012/08/01/171121/aberdeen-1715.tar.gz
[C] out: 169	/data/var/scm/backup/cms02/repos/aberdeen/2012/08/03/171121/aberdeen-1717.tar.gz
[C] out: 168	/data/var/scm/backup/cms02/repos/aberdeen/2012/07/31/171120/aberdeen-1715.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/repos/data/2012/08/02/171121/data-23.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/repos/data/2012/08/01/171121/data-23.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/repos/data/2012/08/03/171121/data-23.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/repos/data/2012/07/31/171120/data-23.tar.gz
[C] out: 5	/data/var/scm/backup/cms02/repos/heprez/2012/08/02/171121/heprez-816.tar.gz
[C] out: 5	/data/var/scm/backup/cms02/repos/heprez/2012/08/01/171121/heprez-816.tar.gz
[C] out: 5	/data/var/scm/backup/cms02/repos/heprez/2012/08/03/171121/heprez-816.tar.gz
[C] out: 5	/data/var/scm/backup/cms02/repos/heprez/2012/07/31/171120/heprez-816.tar.gz
[C] out: 4	/data/var/scm/backup/cms02/repos/newtest/2012/08/02/171121/newtest-20.tar.gz
[C] out: 4	/data/var/scm/backup/cms02/repos/newtest/2012/08/01/171121/newtest-20.tar.gz
[C] out: 4	/data/var/scm/backup/cms02/repos/newtest/2012/08/03/171121/newtest-20.tar.gz
[C] out: 4	/data/var/scm/backup/cms02/repos/newtest/2012/07/31/171120/newtest-20.tar.gz
[C] out: 10	/data/var/scm/backup/cms02/repos/env/2012/08/02/171121/env-3538.tar.gz
[C] out: 10	/data/var/scm/backup/cms02/repos/env/2012/08/01/171121/env-3538.tar.gz
[C] out: 10	/data/var/scm/backup/cms02/repos/env/2012/08/03/171121/env-3538.tar.gz
[C] out: 10	/data/var/scm/backup/cms02/repos/env/2012/07/31/171120/env-3537.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/repos/tracdev/2012/08/02/171121/tracdev-125.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/repos/tracdev/2012/08/01/171121/tracdev-125.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/repos/tracdev/2012/08/03/171121/tracdev-125.tar.gz
[C] out: 1	/data/var/scm/backup/cms02/repos/tracdev/2012/07/31/171120/tracdev-125.tar.gz

monitor cfg: {'HOST': 'H1',
 'HUB': 'C2R',
 'dbpath': '$LOCAL_BASE/env/scm/scm_backup_monitor.db',
 'email': 'blyth@hep1.phys.ntu.edu.tw',
 'jspath': '$APACHE_HTDOCS/data/scm_backup_monitor_%(node)s.json',
 'select': 'repos/env tracs/env repos/aberdeen tracs/aberdeen repos/tracdev tracs/tracdev repos/heprez tracs/heprez',
 'srvnode': 'cms02'} 
[H1] run: find $SCM_FOLD/backup/cms02 -name '*.gz' -exec du --block-size=1M {} \;
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/tracs/tracdev/2012/07/31/171120/tracdev.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/tracs/tracdev/2012/08/01/171121/tracdev.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/tracs/tracdev/2012/08/03/171121/tracdev.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/tracs/tracdev/2012/08/02/171121/tracdev.tar.gz
[H1] out: 2	/home/hep/blyth/var/scm/backup/cms02/tracs/aberdeen/2012/07/31/171120/aberdeen.tar.gz
[H1] out: 2	/home/hep/blyth/var/scm/backup/cms02/tracs/aberdeen/2012/08/01/171121/aberdeen.tar.gz
[H1] out: 2	/home/hep/blyth/var/scm/backup/cms02/tracs/aberdeen/2012/08/03/171121/aberdeen.tar.gz
[H1] out: 2	/home/hep/blyth/var/scm/backup/cms02/tracs/aberdeen/2012/08/02/171121/aberdeen.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/tracs/data/2012/05/01/130104/data.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/tracs/data/2012/04/29/130108/data.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/tracs/data/2012/04/28/130107/data.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/tracs/data/2012/04/30/130103/data.tar.gz
[H1] out: 52	/home/hep/blyth/var/scm/backup/cms02/tracs/env/2012/07/31/171120/env.tar.gz
[H1] out: 52	/home/hep/blyth/var/scm/backup/cms02/tracs/env/2012/08/01/171121/env.tar.gz
[H1] out: 52	/home/hep/blyth/var/scm/backup/cms02/tracs/env/2012/08/03/171121/env.tar.gz
[H1] out: 52	/home/hep/blyth/var/scm/backup/cms02/tracs/env/2012/08/02/171121/env.tar.gz
[H1] out: 9	/home/hep/blyth/var/scm/backup/cms02/tracs/heprez/2012/07/31/171120/heprez.tar.gz
[H1] out: 9	/home/hep/blyth/var/scm/backup/cms02/tracs/heprez/2012/08/01/171121/heprez.tar.gz
[H1] out: 9	/home/hep/blyth/var/scm/backup/cms02/tracs/heprez/2012/08/03/171121/heprez.tar.gz
[H1] out: 9	/home/hep/blyth/var/scm/backup/cms02/tracs/heprez/2012/08/02/171121/heprez.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/tracs/newtest/2012/07/31/171120/newtest.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/tracs/newtest/2012/08/01/171121/newtest.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/tracs/newtest/2012/08/03/171121/newtest.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/tracs/newtest/2012/08/02/171121/newtest.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/folders/svnsetup/2012/07/31/171120/svnsetup.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/folders/svnsetup/2012/08/01/171121/svnsetup.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/folders/svnsetup/2012/08/03/171121/svnsetup.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/folders/svnsetup/2012/08/02/171121/svnsetup.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/repos/tracdev/2012/07/31/171120/tracdev-125.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/repos/tracdev/2012/08/01/171121/tracdev-125.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/repos/tracdev/2012/08/03/171121/tracdev-125.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/repos/tracdev/2012/08/02/171121/tracdev-125.tar.gz
[H1] out: 168	/home/hep/blyth/var/scm/backup/cms02/repos/aberdeen/2012/07/31/171120/aberdeen-1715.tar.gz
[H1] out: 168	/home/hep/blyth/var/scm/backup/cms02/repos/aberdeen/2012/08/01/171121/aberdeen-1715.tar.gz
[H1] out: 169	/home/hep/blyth/var/scm/backup/cms02/repos/aberdeen/2012/08/03/171121/aberdeen-1717.tar.gz
[H1] out: 168	/home/hep/blyth/var/scm/backup/cms02/repos/aberdeen/2012/08/02/171121/aberdeen-1715.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/repos/data/2012/07/31/171120/data-23.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/repos/data/2012/08/01/171121/data-23.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/repos/data/2012/08/03/171121/data-23.tar.gz
[H1] out: 1	/home/hep/blyth/var/scm/backup/cms02/repos/data/2012/08/02/171121/data-23.tar.gz
[H1] out: 10	/home/hep/blyth/var/scm/backup/cms02/repos/env/2012/07/31/171120/env-3537.tar.gz
[H1] out: 10	/home/hep/blyth/var/scm/backup/cms02/repos/env/2012/08/01/171121/env-3538.tar.gz
[H1] out: 10	/home/hep/blyth/var/scm/backup/cms02/repos/env/2012/08/03/171121/env-3538.tar.gz
[H1] out: 10	/home/hep/blyth/var/scm/backup/cms02/repos/env/2012/08/02/171121/env-3538.tar.gz
[H1] out: 5	/home/hep/blyth/var/scm/backup/cms02/repos/heprez/2012/07/31/171120/heprez-816.tar.gz
[H1] out: 5	/home/hep/blyth/var/scm/backup/cms02/repos/heprez/2012/08/01/171121/heprez-816.tar.gz
[H1] out: 5	/home/hep/blyth/var/scm/backup/cms02/repos/heprez/2012/08/03/171121/heprez-816.tar.gz
[H1] out: 5	/home/hep/blyth/var/scm/backup/cms02/repos/heprez/2012/08/02/171121/heprez-816.tar.gz
[H1] out: 4	/home/hep/blyth/var/scm/backup/cms02/repos/newtest/2012/07/31/171120/newtest-20.tar.gz
[H1] out: 4	/home/hep/blyth/var/scm/backup/cms02/repos/newtest/2012/08/01/171121/newtest-20.tar.gz
[H1] out: 4	/home/hep/blyth/var/scm/backup/cms02/repos/newtest/2012/08/03/171121/newtest-20.tar.gz
[H1] out: 4	/home/hep/blyth/var/scm/backup/cms02/repos/newtest/2012/08/02/171121/newtest-20.tar.gz

Disconnecting from blyth@140.112.101.41... done.
Disconnecting from blyth@140.112.101.190... done.
hosts: ['C', 'H1'] 


.. include:: /sphinxext/roles.txt

NTU (hub C2)
-------------
  
+-------------+-------------+------------+------------+----------------+
| node        | nok         | nwarn      | nalarm     | status         |
+=============+=============+============+============+================+
| :alarm:`C`  | :alarm:`12` | :alarm:`0` | :alarm:`1` | :alarm:`alarm` |
+-------------+-------------+------------+------------+----------------+
| :alarm:`H1` | :alarm:`12` | :alarm:`0` | :alarm:`1` | :alarm:`alarm` |
+-------------+-------------+------------+------------+----------------+



C
~~~~~~~~~

+---------------------+---------------+-------+------------------------+---------------------+
| name                | ltime         | lsize | ldays                  | ldate               |
+=====================+===============+=======+========================+=====================+
| tracs/aberdeen      | 1344013881000 | 20.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| tracs/heprez        | 1344013881000 | 90.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| tracs/newtest       | 1344013881000 | 10.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| tracs/env           | 1344013881000 | 52.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| tracs/tracdev       | 1344013881000 | 10.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| folders/svnsetup    | 1344013881000 | 10.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| repos/aberdeen      | 1344013881000 | 169.0 | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| repos/data          | 1344013881000 | 10.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| repos/heprez        | 1344013881000 | 50.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| repos/newtest       | 1344013881000 | 40.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| repos/env           | 1344013881000 | 100.0 | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| repos/tracdev       | 1344013881000 | 10.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| :alarm:`tracs/data` | 1335877264000 | 10.0  | :alarm:`97.9591782407` | 2012-05-01 13:01:04 |
+---------------------+---------------+-------+------------------------+---------------------+



* `scm_backup_monitor_C.json </data/scm_backup_monitor_C.json>`_

.. stockchart:: /data/scm_backup_monitor_C.json container_C

        
H1
~~~~~~~~~

+---------------------+---------------+-------+------------------------+---------------------+
| name                | ltime         | lsize | ldays                  | ldate               |
+=====================+===============+=======+========================+=====================+
| tracs/tracdev       | 1344013881000 | 10.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| tracs/aberdeen      | 1344013881000 | 20.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| tracs/env           | 1344013881000 | 52.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| tracs/heprez        | 1344013881000 | 90.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| tracs/newtest       | 1344013881000 | 10.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| folders/svnsetup    | 1344013881000 | 10.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| repos/tracdev       | 1344013881000 | 10.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| repos/aberdeen      | 1344013881000 | 169.0 | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| repos/data          | 1344013881000 | 10.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| repos/env           | 1344013881000 | 100.0 | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| repos/heprez        | 1344013881000 | 50.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| repos/newtest       | 1344013881000 | 40.0  | 3.78537037037          | 2012-08-03 17:11:21 |
+---------------------+---------------+-------+------------------------+---------------------+
| :alarm:`tracs/data` | 1335877264000 | 10.0  | :alarm:`97.9591782407` | 2012-05-01 13:01:04 |
+---------------------+---------------+-------+------------------------+---------------------+



* `scm_backup_monitor_H1.json </data/scm_backup_monitor_H1.json>`_

.. stockchart:: /data/scm_backup_monitor_H1.json container_H1

        
