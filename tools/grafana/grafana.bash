
grafana-vi(){ vi $BASH_SOURCE ; }
grafana-env(){ echo -n ; }

grafana-usage-almalinux(){ cat << EOU

http://localhost:3000
login with uw/pw:admin/______e   ## changed default pw of admin

EOU
}




grafana-start-almalinux(){ cat << EOS

* https://grafana.com/docs/grafana/latest/setup-grafana/start-restart-grafana/

[blyth@localhost ~]$ sudo systemctl daemon-reload
A[blyth@localhost ~]$ sudo systemctl start grafana-server
A[blyth@localhost ~]$ sudo systemctl status grafana-server


EOS
}

grafana-install-almalinux(){ cat << EOI

* https://grafana.com/docs/grafana/latest/setup-grafana/installation/redhat-rhel-fedora/

wget -q -O gpg.key https://rpm.grafana.com/gpg.key
sudo rpm --import gpg.key


Create /etc/yum.repos.d/grafana.repo::

    [grafana]
    name=grafana
    baseurl=https://rpm.grafana.com
    repo_gpgcheck=1
    enabled=1
    gpgcheck=1
    gpgkey=https://rpm.grafana.com/gpg.key
    sslverify=1
    sslcacert=/etc/pki/tls/certs/ca-bundle.crt


sudo dnf install grafana-enterprise



Grafana Sqlite

* https://grafana.com/grafana/plugins/frser-sqlite-datasource/

* https://github.com/fr-ser/grafana-sqlite-datasource?utm_source=grafana_add_ds


I want to use Grafana or similar to present software metrics persisted into
sqlite3 database ? How to do that ? What sort of tables do I need to prepare to
set this up ?

::

    A[blyth@localhost ~]$ sudo grafana cli plugins install frser-sqlite-datasource
    [sudo] password for blyth: 
    INFO [06-26|11:15:20] Starting Grafana                         logger=settings version=13.1.0 commit=b309c9bb3b81a748c3a75289236a27309ed2566a branch=HEAD compiled=2026-06-23T15:16:42+08:00
    INFO [06-26|11:15:20] Unified migration configs enforced       logger=settings storage_type=unified target=[all]
    INFO [06-26|11:15:20] Enforcing mode 5 for resource in unified storage logger=settings resource=playlists.playlist.grafana.app
    INFO [06-26|11:15:20] Enforcing mode 5 for resource in unified storage logger=settings resource=folders.folder.grafana.app
    INFO [06-26|11:15:20] Enforcing mode 5 for resource in unified storage logger=settings resource=dashboards.dashboard.grafana.app
    INFO [06-26|11:15:20] Config loaded from                       logger=settings file=/usr/share/grafana/conf/defaults.ini
    INFO [06-26|11:15:20] Config loaded from                       logger=settings file=/etc/grafana/grafana.ini
    INFO [06-26|11:15:20] Config overridden from command line      logger=settings arg="default.paths.data=/var/lib/grafana"
    INFO [06-26|11:15:20] Config overridden from command line      logger=settings arg="default.paths.logs=/var/log/grafana"
    INFO [06-26|11:15:20] Config overridden from command line      logger=settings arg="default.paths.plugins=/var/lib/grafana/plugins"
    INFO [06-26|11:15:20] Config overridden from command line      logger=settings arg="default.paths.provisioning=/etc/grafana/provisioning"
    INFO [06-26|11:15:20] Target                                   logger=settings target=[all]
    INFO [06-26|11:15:20] Path Home                                logger=settings path=/usr/share/grafana
    INFO [06-26|11:15:20] Path Data                                logger=settings path=/var/lib/grafana
    INFO [06-26|11:15:20] Path Logs                                logger=settings path=/var/log/grafana
    INFO [06-26|11:15:20] Path Plugins                             logger=settings path="[/var/lib/grafana/plugins /usr/share/grafana/data/plugins-bundled]"
    INFO [06-26|11:15:20] Path Provisioning                        logger=settings path=/etc/grafana/provisioning
    INFO [06-26|11:15:20] App mode production                      logger=settings
    INFO [06-26|11:15:20] Starting Grafana                         logger=settings version=13.1.0 commit=b309c9bb3b81a748c3a75289236a27309ed2566a branch=HEAD compiled=2026-06-23T15:16:42+08:00
    INFO [06-26|11:15:20] Unified migration configs enforced       logger=settings storage_type=unified target=[all]
    INFO [06-26|11:15:20] Enforcing mode 5 for resource in unified storage logger=settings resource=folders.folder.grafana.app
    INFO [06-26|11:15:20] Enforcing mode 5 for resource in unified storage logger=settings resource=dashboards.dashboard.grafana.app
    INFO [06-26|11:15:20] Enforcing mode 5 for resource in unified storage logger=settings resource=playlists.playlist.grafana.app
    INFO [06-26|11:15:20] Config loaded from                       logger=settings file=/usr/share/grafana/conf/defaults.ini
    INFO [06-26|11:15:20] Config loaded from                       logger=settings file=/etc/grafana/grafana.ini
    INFO [06-26|11:15:20] Config overridden from command line      logger=settings arg="default.paths.data=/var/lib/grafana"
    INFO [06-26|11:15:20] Config overridden from command line      logger=settings arg="default.paths.logs=/var/log/grafana"
    INFO [06-26|11:15:20] Config overridden from command line      logger=settings arg="default.paths.plugins=/var/lib/grafana/plugins"
    INFO [06-26|11:15:20] Config overridden from command line      logger=settings arg="default.paths.provisioning=/etc/grafana/provisioning"
    INFO [06-26|11:15:20] Target                                   logger=settings target=[all]
    INFO [06-26|11:15:20] Path Home                                logger=settings path=/usr/share/grafana
    INFO [06-26|11:15:20] Path Data                                logger=settings path=/var/lib/grafana
    INFO [06-26|11:15:20] Path Logs                                logger=settings path=/var/log/grafana
    INFO [06-26|11:15:20] Path Plugins                             logger=settings path="[/var/lib/grafana/plugins /usr/share/grafana/data/plugins-bundled]"
    INFO [06-26|11:15:20] Path Provisioning                        logger=settings path=/etc/grafana/provisioning
    INFO [06-26|11:15:20] App mode production                      logger=settings

    ✔ Downloaded and extracted frser-sqlite-datasource v4.0.6 zip successfully to /var/lib/grafana/plugins/frser-sqlite-datasource

    Please restart Grafana after installing or removing plugins. Refer to Grafana documentation for instructions if necessary.




    A[blyth@localhost ~]$ sudo systemctl restart grafana-server



SELECT 
  time,
  hostname as metric,
  cpu_utilization
FROM device_telemetry
WHERE $__timeFilter(time)
ORDER BY time ASC;


SELECT CAST(strftime('%s', 'now', '-1 minute') as INTEGER) as time, 4 as value 
WHERE time >= $__from / 1000 and time < $__to / 1000


sqlite> select cast(strftime('%s',t) as INTEGER) as time from device_telemetry ;
+------------+
|    time    |
+------------+
| 1782454243 |
| 1782454543 |
| 1782454843 |
| 1782455143 |
| 1782455443 |
| 1782454243 |
| 1782454543 |
| 1782454843 |
| 1782455143 |
| 1782455443 |
+------------+


sqlite> select cast(strftime('%s',t) as INTEGER) as time, hostname as metric, cpu_utilization as value from device_telemetry order by time asc ;
+------------+---------+-------+
|    time    | metric  | value |
+------------+---------+-------+
| 1782454243 | node-01 | 42.5  |
| 1782454243 | node-02 | 12.1  |
| 1782454543 | node-01 | 48.1  |
| 1782454543 | node-02 | 14.5  |
| 1782454843 | node-01 | 55.0  |
| 1782454843 | node-02 | 11.2  |
| 1782455143 | node-01 | 63.8  |
| 1782455143 | node-02 | 18.9  |
| 1782455443 | node-01 | 51.2  |
| 1782455443 | node-02 | 15.4  |
+------------+---------+-------+


SELECT CAST(strftime('%s', t) as INTEGER) as time, cpu_utilization as value 
FROM device_telemetry
WHERE time >= $__from / 1000 and time < $__to / 1000
order by time asc


grafana time series panel requires
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. time
2. value
3. metric [text column used for grouping]


grafana "wide layout query" does not include the metric but plots all the selected numerical columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SELECT 
  CAST(strftime('%s', t) as INTEGER) as time, 
  cpu_utilization as "Node 1 CPU",
  memory_utilization as "Node 1 Memory"
FROM device_telemetry
WHERE time >= $__from / 1000 and time < $__to / 1000;






EOI
}

