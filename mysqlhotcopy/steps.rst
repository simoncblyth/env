Next Steps
============

The mysqlhotcopy tarball for the recovered "channelquality_db"
was prepared using a mysql server on belle1.nuu.edu.tw
installed for this purpose with precisely the same mysql
version 5.0.45-community as that used on dybdb1.ihep.ac.cn.

Next steps summary for the parts I am aware of:

   1. Qiumei extracts "channelquality_db" hotcopy tarball onto dybdb1
      http://belle7.nuu.edu.tw/e/mysqlhotcopy/mysql_repair_table_live/#dybdb1-extraction-instructions-for-qiumei
      [this will bring free space on the server /data down to ~9G]

      Testing the extraction of the 2.3G tarball onto belle7 took
      less than 5 min, creating a ~9.2G database directory.
      Other approaches such as digesting a mysqldump or CSV files were
      vetoed as they took ~40-70 min during which time the server was
      unresponsive.

   2. Simon, Gaosong test that "channelquality_db" is equivalent
      to "tmp_ligs_offline_db" up to SEQNO 323573.
      To facilitate comparison will instruct Qiumei to do a "repair table"
      to DqChannelStatus on the server [early next week]

   3. Qiumei installs cron invoked disk space monitoring script on the server
      http://belle7.nuu.edu.tw/e/db/valmon/

   4. Once "channelquality_db" is validated Qiumei can drop "tmp_ligs_offline_db"
      to free ~10G disk space on the server
      [back to ~19G free on the server]

   5. Miao/Gaosong re-fill into "channelquality_db" the lost entries with SEQNO > 323573
      [estimated 2 days]
      [guess down to ~17G]

   6. Simon runs the compression script creating DqChannelPacked+Vld
      [running time was 26hrs up to SEQNO 323000, so estimate ~1 day to
      extend that to cover the KUP re-filling]

      The packed tables are a factor of 125 times smaller than the profligate DqChannelStatus,
      so mysqldump loading can be used to propagate the new table into offline_db

   7. Liang load mysqldump into  offline_db.DqChannelPacked+Vld

   8. Brett tests service reading from offline_db.DqChannelPacked

   9. Craig, Weidong provide solution for tight disk space situation on the server.
      A new disk ? A new server ?

      Operating with such a tight disk space situation on our primary DB server
      is asking for trouble.

