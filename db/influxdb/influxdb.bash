# === func-gen- : db/influxdb/influxdb fgp db/influxdb/influxdb.bash fgn influxdb fgh db/influxdb
influxdb-src(){      echo db/influxdb/influxdb.bash ; }
influxdb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(influxdb-src)} ; }
influxdb-vi(){       vi $(influxdb-source) ; }
influxdb-env(){      elocal- ; }
influxdb-usage(){ cat << EOU


InfluxDB
==========


* https://github.com/influxdata/influxdb
* https://en.wikipedia.org/wiki/InfluxDB
* https://news.ycombinator.com/item?id=11262318

Alternatives
--------------

* :google:`influxdb prometheus graphite opentsdb`
* https://prometheus.io/
* http://opentsdb.net/overview.html

graphite
      round-robin based, so expects samples at regular intervals

Comparisons
-------------

* https://prometheus.io/docs/introduction/comparison/

At its core, InfluxDB stores timestamped events with full metadata (key-value
pairs) attached to each event / row. Prometheus stores only numeric time series
and stores metadata for each time series exactly once, and then continues to
simply append timestamped samples for that existing metadata entry.


Can time series DB work for detector calibrations ?
------------------------------------------------------

Monitoring of web services seems to be the 
canonical use case of most existing time series DB. 
A calibration DB requires the recording of the state of a detector 
and being able to return to that state, and override that state via versioning.  
It is not clear whether the time series DB can be made to do this, or if
it is wise to use a product in a way it was not really intended for.

Perhaps something like an inventory management database might be
closer to the requirements of a calibration DB.


Searching
-----------

* :google:`instrument calibration database`



:google:`open source instrument calibration database`
--------------------------------------------------------

Open source software for visualization and quality control of 
continuous hydrologic and water quality sensor data

* http://www.sciencedirect.com/science/article/pii/S1364815215001115
* ~/dbi_refs/odm_tools.pdf


ODM2
~~~~~

* https://github.com/ODM2/ODM2


NASA mission instrument caldb
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_91_001/orig/cal_gen_91_001.pdf


What is really needed ?
------------------------

* solution providing usage workflow as well as technical implementation 





EOU
}
influxdb-dir(){ echo $(local-base)/env/db/influxdb/db/influxdb-influxdb ; }
influxdb-cd(){  cd $(influxdb-dir); }
influxdb-mate(){ mate $(influxdb-dir) ; }
influxdb-get(){
   local dir=$(dirname $(influxdb-dir)) &&  mkdir -p $dir && cd $dir

}
