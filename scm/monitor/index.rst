
:modified: 2012-07-10 08:42:54+00:00
:tags: Sphinx


Monitor
=========

Plots monitoring tarball sizes and counts 
for Trac and Subversion instances that hail from **hub** servers 
at various institutions. Typically plots for multiple backup nodes are 
listed for each hub server.

.. toctree::
    :glob:

    *

How it works
---------------

Sphinx ``.. raw:: html`` directives are used to embed javascript (use show source on right to see this) and a single **div** into the html built version of this 
page. On page load the javascript runs an ajax query to pull in the plot data and options from a static JSON files for each remote node residing in `</data/>`_. These 
static files are created by the ``scm-backup-monitor`` which using **fabric** to gather info from remote nodes and updates an SQLite DB.


If had large numbers of plots to render, it would be silly to re-render in browser
for quntities that are only updated daily.  But that is what this is doing.  

* can the plot be rendered as an image on the server ? allow this to be done once only 

   
Setup auto-monitoring on dayabay.ihep.ac.cn
----------------------------------------------

To test the auto-monitoring script requires:

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


To manually test operation run the `monitor.py` script as shown below::

         mkdir -p /var/www/html/data    ## create output dir for json plot data
         cd ~/env/scm
         ~blyth/local/python/Python-2.5.6/bin/python monitor.py      
                ## have to use my python to pickup needed modules : fabric, converter, ...



