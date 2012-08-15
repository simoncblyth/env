
:modified: 2012-07-10 08:42:54+00:00
:tags: Sphinx


Monitor
=========

.. contents:: :local:

Plots monitoring tarball sizes and counts for Trac and Subversion instances 
that reside on **hub** servers at various institutions. Typically each hub has multiple backup nodes 
with the corresponding plots being presented by hub.

.. toctree::
    :glob:

    *

Setup
-------

To setup the auto-monitoring script on a new node requires the below
steps as described on this page.

#. configuring
#. manual testing 
#. automation


How the plotting work
-----------------------

A globbed toctree directive is used to pull in all rst pages from `env/scm/monitor/`
These pages use the sphinx extension `env.sphinxext.stockchart` to embed raw html
containing javascript and a single div element into the html output derived from
the reStructuredText source (use show source to see this).

On page load the javascript runs an ajax query to pull in the plot data and options from 
static JSON files for each remote node residing in `</data/>`_. 
These static files are created by the ``scm-backup-monitor`` which uses **fabric** 
to gather info from remote nodes and updates an SQLite DB.

.. note:: the plots are rendered on the client, this approach as it stands is thus limited to small numbers of things to monitor

 
Configuring the monitoring 
-----------------------------

#. update env checkout in **root** home directory
#. copy two config files into **root** home::

         cd 
         cp ~blyth/.scm_monitor.cnf .
         cp ~blyth/.libfab.cnf .

The ZZ section of `.scm_monitor.cnf` configures where
a database file and output json files are stored.::

	[dayabay] /home/blyth/e > cat ~blyth/.scm_monitor.cnf 

	[ZZ]
	srvnode = dayabay
	dbpath = $LOCAL_BASE/env/scm/scm_backup_monitor.db
	jspath = $APACHE_HTDOCS/data/scm_backup_monitor_%(node)s.json
	select = svn/dybsvn tracs/dybsvn svn/dybaux tracs/dybaux
	reporturl = http://dayabay.ihep.ac.cn:8080/e/scm/monitor/%(srvnode)s/

	[WW]
	message = "query the cms02 hub backup on cms01, as I do not have access to the SDU one "
	srvnode = cms02
	dbpath = $LOCAL_BASE/env/scm/scm_backup_monitor.db
	jspath = $HOME/local/nginx/html/data/scm_backup_monitor_%(node)s.json
	select = repos/env tracs/env repos/aberdeen tracs/aberdeen repos/tracdev tracs/tracdev repos/heprez tracs/heprez
	reporturl = http://dayabay.ihep.ac.cn:8080/e/scm/monitor/%(srvnode)s/


The `ZZ = SDU` in the **HUB** section of `.libfab.cnf`  configures 
the node tags of remote nodes on which to look for tarballs, using the **fabric** python module
to run commands over ssh::

	[dayabay] /home/blyth/e > cat ~blyth/.libfab.cnf

	[HUB]
	ZZ = SDU
	G = Z9:229
	C2 = C H1
	WW = C

	[ENV]
	verbose = True
	timeout = 2


Manual Testing
----------------

To manually test operation run the `monitor.py` script as shown below::

         mkdir -p /var/www/html/data    ## create output dir for json plot data if not already existing

         ## have to use local python to pickup needed modules : fabric, converter, ...

         env-                
         scm-backup-      
                          ## setup environment, eg APACHE_HTDOCS, LOCAL_BASE referred to in the config

         export LOCAL_PYTHON=/home/blyth/local/python/Python-2.5.6
         export LD_LIBRARY_PATH=$LOCAL_PYTHON/lib 
         export PATH=$LOCAL_PYTHON/bin:$PATH
 
         ~/env/scm/monitor.py      



Automation
------------

The running of the `monitor.py` is done as part of the standard `scm-backup-nightly` function which can
be invoked via a crontab line such as the below::


	[root@cms02 env]# crontab -l
	SHELL = /bin/bash
	16 18 * * *  ( export HOME=/root ; export NODE=cms02 ; export MAILTO=blyth@hep1.phys.ntu.edu.tw ; export ENV_HOME=/home/blyth/env ; . /home/blyth/env/env.bash ; env-  ; scm-backup- ; scm-backup-nightly ) >  /var/scm/log/scm-backup-nightly-$(date +"\%a").log 2>&1


When adding a new node, the `scm-backup-nightly` needs updating to add the running of the monitor script.::


	[root@cms02 env]# scm-backup-
	[root@cms02 env]# t scm-backup-nightly
	scm-backup-nightly is a function
	scm-backup-nightly () 
	{ 
	    local msg="=== $FUNCNAME :";
	    echo;
	    echo $msg $(date) @@@ scm-backup-checkscp;
	    scm-backup-checkscp;
	    echo;
	    echo $msg $(date) @@@ scm-backup-all;
	    scm-backup-all;
	    echo;
	    echo $msg $(date) @@@ scm-backup-rsync ... performing transfers that i control;
	    scm-backup-rsync;
	    echo;
	    echo $msg $(date) @@@ scm-backup-parasitic ... monitoring transfers that i do not control... i just receive the tarballs;
	    case $NODE_TAG in 
		C2 | C2R)
		    scm-backup-parasitic ZZ C
		;;
		*)
		    echo $msg no parasitic monitoring is configured on NODE_TAG $NODE_TAG
		;;
	    esac;
	    echo;
	    echo $msg $(date) @@@ scm-backup-monitor ... fabric remote tarball checking;
	    case $NODE_TAG in 
		G)
		    scm-backup-monitor- G
		;;
		C2 | C2R)
		    scm-backup-monitor- C2
		;;
		*)
		    echo $msg scm-backup-monitor not yet implemented on $NODE_TAG
		;;
	    esac;
	    echo;
	    echo $msg $(date) @@@ scm-backup-nightly ... completed;
	    echo
	}
	[root@cms02 env]# 



Improvement Ideas
-------------------


The inclusion of inline RST status tables (indicating what caused problems) 
in the output means that the html must be updated after each monitoring run.  
This is done by `scm-backup-monitor-` and is a potential source of fragility as documentation is
updated for the project.
Prior to adding these RST tables it was not necessary to update the html every time as the plots 
are dynamically included by the javascript loading the json on page load. Possibly a javascript
library for rendering tables might avoid the need to regenerate the html every time allowing the 
table content to be pulled in from static json files.

Somewhat conversly it is limiting that precisely the same plots are re-rendered 
for every client in the browser. It would be more efficient to render plots once when data is updated 
and have the clients simply load an image.  Doing this in a manner that benefits from 
the javascript library code is non trivial. Google V8 Javascript engine tied with node.js for 
javascript on the server is a likely way this might be done in future.   

  * http://blog.davidpadbury.com/2010/10/03/using-nodejs-to-render-js-charts-on-server/

For this to work smoothly needs effort from js library authors, basically it is porting the 
js to a non-standard **browser** that can live on the server.

  * http://highslide.com/forum/viewtopic.php?f=12&t=16380
     
     * some noise indicating might happen by version 3.0 of highcharts

  * http://www.highcharts.com/support/roadmap
 
     * not in offical roadmap
     









