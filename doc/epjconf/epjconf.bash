epjconf-source(){   echo ${BASH_SOURCE} ; }
epjconf-vi(){   vi $(epjconf-source) ; }
epjconf-env(){  elocal- ; }
epjconf-usage(){ cat << EOU

EPJ Web Of Conferences
=========================

* https://www.epj-conferences.org/
* https://www.epj-conferences.org/for-authors
* https://www.epj-conferences.org/doc_journal/woc/epjconf_editorial_guidelines.pdf

Before submitting print-ready PDF files, please check the quality of the final
PDF documents at 

* http://pdf-analyser.edpsciences.org

  Check your final PDF documents here to verify that all fonts used in your
  document are embedded and if the quality of the images is good enough.
  The PDF document is not saved in our server after being checked.


Setup
-------

* manually copied contents of ftp://ftp.edpsciences.org/pub/web-conf  
  (which opened in Finder very slowly) to /usr/local/epjconf/web-conf


CHEP 2016
----------

See ioproc- the predecessor of epjconf-

CHEP 2018
------------

* The CHEP 2018 proceedings will be published in the EPJ Web of Conferences

proceeding preparation guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://chep2018.org/proceedings

  * 8 pages, single-column format, submit in PDF (but use the latex templates)

painful web interface
~~~~~~~~~~~~~~~~~~~~~~~~~

* https://saga.edpsciences.org/topic/epjconf/400

* Your submission has been successfully registered with the reference: epjconf182175. 
  You can now upload the file(s) pertaning to this submission.


code citations
~~~~~~~~~~~~~~~~~~

* https://indico.cern.ch/event/587955/contributions/3012261/attachments/1683270/2706352/Katz_2018.07.09_software_citations_CHEP.pdf

* ~/opticks_refs/Katz_2018.07.09_software_citations_CHEP.pdf

* https://choosealicense.com

* https://guides.github.com/activities/citable-code/

* https://figshare.com

* https://zenodo.org

Bitbucket zenodo ? Not yet 

* https://bitbucket.org/site/master/issues/9448/provide-api-for-adding-hooks
* https://stackoverflow.com/questions/33882099/how-to-create-digital-object-identifier-doi-for-bitbucket-repository
 


CHEP2018 proc : how to structure an update proceedings ?
--------------------------------------------------------------

Blyth Simon C 2017 J. Phys.: Conf. Ser. 898 042001

Accelerating navigation in the VecGeom geometry modeller   (2016)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://iopscience.iop.org/article/10.1088/1742-6596/898/7/072032/pdf
* ~/opticks_refs/vecgeom_Wenzel_2017_J._Phys_3A_Conf._Ser._898_072032.pdf 


At the end of the introduction::

    A lot of the effort of the VecGeom project has so far gone into development
    of algorithms for basic geometric entities within component (a) in order to
    enable multi-particle operations using SIMD processing or to improve on
    existing code. This work has been presented previously [10, 11] and continues
    to be improved and extended constantly.

    The present proceeding focuses on our recent developments done for the
    navigation components (b). In section 2, we will describe the work undertaken
    to achieve scalable and SIMD enabled collision detection and location queries.
    In section 3, complementary ideas are presented for speeding up navigation
    algorithms, based on the idea of complete code specialisation. 



[10] Apostolakis J, Brun R, Carminati F, Gheata A and Wenzel S 2014 J. Phys.: Conf. Ser. 513 052038
[11] Apostolakis J et al 2015 J. Phys.: Conf. Ser. 608 012023


EOU
}

# env machinery dir
epjconf-edir(){ echo $(dirname $(epjconf-source)) ; }
epjconf-ecd(){  cd $(epjconf-edir); }

# template dir
epjconf-tdir(){  echo $LOCAL_BASE/epjconf/web-conf ; }
epjconf-tcd(){   cd $(epjconf-tdir); }
epjconf-tdoc(){ open $(epjconf-tdir)/webofc-doc.pdf ; } 

# latex source dir
epjconf-dir(){  echo $(env-home)/doc/epjconf/$(epjconf-confname) ; }
epjconf-cd(){   cd $(epjconf-dir); }
epjconf-confname(){ echo chep2018 ; }
epjconf-texname(){  echo $(epjconf-confname).tex ; }
epjconf-textmpl(){  echo template.tex ; }

epjconf-2016(){ open http://iopscience.iop.org/article/10.1088/1742-6596/898/4/042001/meta ; }

epjconf-init-notes(){ cat << EON

$FUNCNAME
===================

epjconf-tdir    : $(epjconf-tdir)
epjconf-textmpl : $(epjconf-textmpl)
epjconf-dir     : $(epjconf-dir)
epjconf-texname : $(epjconf-texname)

Copies template files from tdir to the repository dir for editing.
Actually now that are using TEXINPUTS envvar there is no need 
to copy the cls file can just use it directly, thus have commented it.

This only needs to be run once for a conference proceedings, 
after doing so remember to commit the sources into the repository.

EON
}

epjconf-init()
{
   epjconf-init-notes
   local msg="=== $FUNCNAME : "
   local iwd=$PWD
   local dir=$(epjconf-dir)
   local tdir=$(epjconf-tdir)
    
   echo $msg $dir 
   mkdir -p $dir && cd $dir

   local texname=$(epjconf-texname)
   local textmpl=$(epjconf-textmpl)

   echo $msg dir $dir 

   if [ -f "$texname" ]; then
       echo $msg texname $texname exists already
   else
       echo $msg copying textmpl $textmpl to texname $texname 
       cp $tdir/$textmpl $texname 
   fi 

   if [ -f "Makefile" ] ; then
       echo $msg Makefile exists already 
   else
       echo $msg writing Makefile
       epjconf-Makefile > Makefile
   fi 

   cd $iwd
}


epjconf-odir(){ echo /tmp/$USER/epjconf ; }
epjconf-opdf(){ echo $(epjconf-odir)/$(epjconf-confname).pdf ; }
epjconf-open(){ open $(epjconf-opdf) ; }

epjconf-figdir(){ echo $HOME/simoncblyth.bitbucket.io ; }

epjconf-texinputs(){ echo .:$(epjconf-tdir):$(epjconf-figdir): ; }

epjconf-texinputs-notes(){ cat << EON

TEXINPUTS envvar
-----------------

* https://tex.stackexchange.com/questions/93712/definition-of-the-texinputs-variable

Double slash at end of dir triggers recursive search ?

   epjconf-figdir  : $(epjconf-figdir) 

For consistency with other sources such as RST slides, the figdir
is set to at top level of ~/simoncblyth.bitbucket.io 
Thus to include images use relative paths starting with env::

    includegraphics[width=5cm,clip]{env/ok/dyb_raytrace_composite_cerenkov_half_half}

EON
}

epjconf-pdflatex()
{
    epjconf-cd
    local odir=$(epjconf-odir)
    mkdir -p $odir 
    TEXINPUTS=$(epjconf-texinputs) pdflatex -output-directory $odir $(epjconf-texname)
}

epjconf--()
{
    epjconf-pdflatex
    epjconf-open
}

epjconf-etex(){ echo $(epjconf-dir)/$(epjconf-confname).tex ; }
epjconf-edit(){ local etex=$(epjconf-etex) ; echo $FUNCNAME etex $etex ; vi $etex $(epjconf-aux) ;  }
epjconf-e(){    epjconf-edit ; }

epjconf-aux(){ cat << EOA
/Users/blyth/env/presentation/opticks_gpu_optical_photon_simulation_jul2018_chep.txt
EOA
}


epjconf-Makefile(){ cat << EOM
# generated by $FUNCNAME $(date)

.PHONY: go default

default: go
	@echo done

go:
	@echo go 
	bash -lc "epjconf- ; epjconf-- " 

EOM
}

epjconf-check-pdf-notes(){ cat << EON

Navigate to $(epjconf-odir) to upload the PDF for checking, 
the web interface lists fonts and their embedded status 
and images with their DPI.  Need at least 300 DPI. 

EON
}
epjconf-check-pdf(){ $FUNCNAME-notes ; open http://pdf-analyser.edpsciences.org ; }
