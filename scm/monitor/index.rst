
:modified: 2012-07-10 08:42:54+00:00
:tags: Sphinx


Monitor
=========

Plots monitoring tarball sizes and counts 
for Trac and Subversion instances that hail from **hub** servers 
at various institutions. Typically plots for multiple backup nodes are 
listed for each hub server.

.. toctree::

    hku
    ihep
    cms02


How it works
---------------

Sphinx ``.. raw:: html`` directives are used to embed javascript (use show source on right to see this) and a single **div** into the html built version of this 
page. On page load the javascript runs an ajax query to pull in the plot data and options from a static JSON files for each remote node residing in `</data/>`_. These 
static files are created by the ``scm-backup-monitor`` which using **fabric** to gather info from remote nodes and updates an SQLite DB.


If had large numbers of plots to render, it would be silly to re-render in browser
for quntities that are only updated daily.  But that is what this is doing.  

* can the plot be rendered as an image on the server ? allow this to be done once only 

More details on mechanicas and dev notes:

.. toctree::

    dev
    


