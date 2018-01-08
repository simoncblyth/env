CQ
====

* http://dayawane.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/DataQuality/DQDump/share/DumpDqPmt.py#L216


* http://dayabay.ihep.ac.cn/tracs/dybsvn/ticket/1389

* vi ~/dybgaudi/DybPython/python/DybPython/dbsrv.py

* https://bitbucket.org/simoncblyth/env/commits/all?page=86

coldcopy : just a file copy with no locking 

* https://bitbucket.org/simoncblyth/env/commits/3ba47d9af6cdc73383b2d31ee156feb3d0cadb65

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Database/Scraper/scripts/CQCatchup.sh


::

    simon:~ blyth$ cd ~/e/mysqlhotcopy/
    simon:mysqlhotcopy blyth$ l
    total 520
    -rw-r--r--  1 blyth  staff       0 Sep  1  2014 __init__.py
    -rwxr-xr-x  1 blyth  staff    1226 Sep  1  2014 cmd.py
    -rw-r--r--  1 blyth  staff    1192 Sep  1  2014 common.py
    -rwxr-xr-x  1 blyth  staff    7593 Sep  1  2014 db.py
    -rwxr-xr-x  1 blyth  staff    1453 Sep  1  2014 dbcf.py
    -rwxr-xr-x  1 blyth  staff    7697 Sep  1  2014 fsutils.py
    -rw-r--r--  1 blyth  staff     153 Sep  1  2014 index.rst
    -rw-r--r--  1 blyth  staff     192 Sep  1  2014 lessons.rst
    -rw-r--r--  1 blyth  staff    7770 Sep  1  2014 locking.rst
    -rw-r--r--  1 blyth  staff   28249 Sep  1  2014 mysql_repair_table.rst
    -rw-r--r--  1 blyth  staff  124700 Sep  1  2014 mysql_repair_table_live.rst
    -rwxr-xr-x  1 blyth  staff   37089 Sep  1  2014 mysqlhotcopy.py
    -rw-r--r--  1 blyth  staff     104 Sep  1  2014 mysqlhotcopy.rst
    -rw-r--r--  1 blyth  staff    2258 Sep  1  2014 steps.rst
    -rwxr-xr-x  1 blyth  staff   14692 Sep  1  2014 tar.py
    simon:mysqlhotcopy blyth$ 





::

    Fri Oct 20 22:13:01 CST 2017
    Continue Logging to /dev/stdout as requested by -l option, logtruncate


    Starting dybinst commands: scrape

    Stage: "scrape"... 

    === ds_gopkn : pwd /dyb/nuwadata/NuWa/NuWa-trunk
    === ds_cdpkd : entering dybgaudi/DybRelease environment pwd /dyb/nuwadata/NuWa/NuWa-trunk
    /dyb/nuwadata/NuWa/NuWa-trunk
    === ds_cdpkd : entering /dyb/nuwadata/NuWa/NuWa-trunk/dybgaudi/Database/Scraper environment pwd /dyb/nuwadata/NuWa/NuWa-trunk
    /dyb/nuwadata/NuWa/NuWa-trunk
    === dybinst-scrape : env prior to exec
    SCRAPER_CFG=/dyb/nuwadata/NuWa/NuWa-trunk/dybgaudi/Database/Scraper/python/Scraper/.scraper.cfg
    === dybinst-scrape : cfg "CQScraper"
    ds_scraper_cmd is a function
    ds_scraper_cmd () 
    { 
        local cfg=$1;
        case $cfg in 
            CQScraper)
                echo "$(which CQScraper.py)"
            ;;
            CQCatchup)
                echo "$(which CQCatchup.sh)"
            ;;
            *)
                echo "$(which scr.py) -s $cfg "
            ;;
        esac
    }
    === dybinst-scrape : exec : /dyb/nuwadata/NuWa/NuWa-trunk/dybgaudi/InstallArea/scripts/CQScraper.py
    2017-10-20 22:13:14,936 Scraper.dq.CQConfig INFO     /dyb/nuwadata/NuWa/NuWa-trunk/dybgaudi/InstallArea/scripts/CQScraper.py
    2017-10-20 22:13:15,850 Scraper.dq.CQScraper INFO     src: CQTable channelquality_db.DqChannelStatus          lastseqno:1695261  
    2017-10-20 22:13:15,951 Scraper.dq.CQScraper INFO     tgt: CQTable offline_db_rw.DqChannelPacked              lastseqno:1694900  
    Traceback (most recent call last):
      File "/dyb/nuwadata/NuWa/NuWa-trunk/dybgaudi/InstallArea/scripts/CQScraper.py", line 4, in <module>
        main()
      File "/dyb/nuwadata/NuWa/NuWa-trunk/dybgaudi/InstallArea/python/Scraper/dq/CQScraper.py", line 318, in main
        baseseqno, recs = scraper.readsrc() 
      File "/dyb/nuwadata/NuWa/NuWa-trunk/dybgaudi/InstallArea/python/Scraper/dq/CQScraper.py", line 190, in wrapper
        res = func(*arg,**kw)
      File "/dyb/nuwadata/NuWa/NuWa-trunk/dybgaudi/InstallArea/python/Scraper/dq/CQScraper.py", line 219, in readsrc
        recs = self.src.getEntries(lastseqno)
      File "/dyb/nuwadata/NuWa/NuWa-trunk/dybgaudi/InstallArea/python/Scraper/dq/CQTable.py", line 109, in getEntries
        assert rec['n'] == rec['row_counter'] == 192, (rec['seqno'],rec['n'], rec['row_counter'] )
    AssertionError: (1695113L, 180L, 180L)




* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Database/Scraper/python/Scraper/dq/CQTable.py
* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Database/Scraper/python/Scraper/dq/CQJudge.py
* https://dev.mysql.com/doc/refman/5.7/en/case.html




