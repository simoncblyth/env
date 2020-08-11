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



July 2020 : Referee Comments
------------------------------

[4]
S. Blyth, EPJ Web Conf. {\bf 214}, 02027 (2019)

[6]
S. Agostinelli, J. Allison, K. Amako, J. Apostolakis, H. Araujo, P. Arce et al., Nucl. Instrum. Methods. Phys. Res. A {\bf 506}, 250 (2003)

[7]
J. Allison, K. Amako, J. Apostolakis, H. Araujo, P. Dubois, M. Asai et al., IEEE Trans Nucl Sci, {\bf 53}, 270 (2006)

[8]
J. Allison, K. Amako, J. Apostolakis, P. Arce, M. Asai, T. Aso et al., Nucl. Instrum. Methods. Phys. Res. A {\bf 835}, 186 (2016)

[9]
S. Parker, J. Bigler, A. Dietrich, H. Friedrich, J. Hoberock et al., ACM Trans. Graph.: Conf. Series {\bf 29}, 66 (2010)

[12]
F. An et al., J. Phys. G. {\bf 43}, 030401 (2016)

[14]
M. Garland, D.B. Kirk, Commun. ACM {\bf 53}(11), 58 (2010)

[16]
S. Van der Walt, S. Colbert, G. Varoquaux, Comput. Sci. Eng. {\bf 13}, 22 (2011)







CHEP 2019 : What is new ?
---------------------------

* focus of 2019 : was making Opticks benefit from the performance
  boost provided by RT cores

* performance figures
* now have full JUNO geometry measurements, not extrapolations

  * can make more specific statements and name GPUs

* validation numbers are available 
* Turing no longer recent


* https://simoncblyth.bitbucket.io/env/presentation/opticks_gpu_optical_photon_simulation_nov2019_chep.html


CHEP 2019 proceedings
-----------------------

* https://simoncblyth.bitbucket.io

* https://indico.cern.ch/event/773049/page/19236-proceedings

* Templates to prepare your paper are available for LaTeX and Word (with instructions)
* https://indico.cern.ch/event/773049/attachments/1926162/3253016/woc_1col.pdf
* Please use the single column format
* Suggested page lengths are 6 pages for parallel contributions, 8 pages for plenary contributions, excluding references
* Please submit your paper by 14 March 2020


* ~/Downloads/epj-woc-latex copied into /usr/local/epjconf BUT no changes
  compared to last year, so just start from previous epjconf proceedings 
  copied into /Users/blyth/env/doc/epjconf/chep2019


No relevant changes::

    epsilon:epjconf blyth$ pwd
    /usr/local/epjconf
    epsilon:epjconf blyth$ mv ~/chep2019/epj-woc-latex/ .
    epsilon:epjconf blyth$ l
    total 0
    drwx------@ 9 blyth  staff  288 Feb 24 13:48 epj-woc-latex
    drwxr-xr-x  8 blyth  staff  256 Apr  8  2019 web-conf
    epsilon:epjconf blyth$ diff epj-woc-latex/template.tex web-conf/template.tex 
    epsilon:epjconf blyth$ 
    epsilon:epjconf blyth$ 
    epsilon:epjconf blyth$ l web-conf/
    total 376
    -rw-------@ 1 blyth  staff  44064 Oct  1  2018 woc.bst
    -rw-------@ 1 blyth  staff  22879 Oct  1  2018 webofc.cls
    -rw-------@ 1 blyth  staff  88277 Oct  1  2018 webofc-doc.pdf
    -rw-------@ 1 blyth  staff   4055 Oct  1  2018 template.tex
    -rw-------@ 1 blyth  staff   1489 Oct  1  2018 readme.txt
    -rw-------@ 1 blyth  staff  20849 Oct  1  2018 additional-styles.tar.gz
    epsilon:epjconf blyth$ l epj-woc-latex/
    total 384
    -rwxr-xr-x@ 1 blyth  staff  44064 Nov  7 12:21 woc.bst
    -rwxr-xr-x@ 1 blyth  staff  22879 Nov  7 12:20 webofc.cls
    -rwxr-xr-x@ 1 blyth  staff   4059 Nov  7 12:20 template_twocolumn.tex
    -rwxr-xr-x@ 1 blyth  staff   4055 Nov  7 12:20 template.tex
    -rwxr-xr-x@ 1 blyth  staff   1489 Nov  7 12:20 readme.txt
    -rwxr-xr-x@ 1 blyth  staff  20849 Nov  7 12:19 additional-styles.tar.gz
    -rwxr-xr-x@ 1 blyth  staff  88277 Nov  6 12:23 webofc-doc.pdf
    epsilon:epjconf blyth$ 

    epsilon:epjconf blyth$ diff -r --brief web-conf epj-woc-latex
    Only in epj-woc-latex: template_twocolumn.tex
    epsilon:epjconf blyth$ 





Sep 17, 2019 : CHEP2018 proceedings published
------------------------------------------------

* https://www.epj-conferences.org/articles/epjconf/abs/2019/19/contents/contents.html#section_10.1051/epjconf/201921402001


April 8 : uploaded response to comments and amended draft
-----------------------------------------------------------

* https://saga.edpsciences.org/article/epjconf/epjconf182175/sheet/sheet/


April 7 : Aborted attempt to use bibtex
---------------------------------------------

The woc.bst style file that epjconf instructs
to be used is mangling authors.

I wanted to use bibtex for cite sets, but it looks too much hassle.

* https://texfaq.org/FAQ-mcite

* https://tex.stackexchange.com/questions/171175/biblatex-mcite-add-arbitrary-text-in-references-with-subentries

* http://mirrors.ibiblio.org/CTAN/macros/latex/exptl/biblatex/doc/biblatex.pdf


mcite fails with cites in captions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://tex.stackexchange.com/questions/174275/protect-with-cite-inside-caption

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

# output dir
epjconf-odir(){ echo /tmp/$USER/epjconf ; }
epjconf-ocd(){ 
   local odir=$(epjconf-odir) 
   [ ! -d "$odir" ] && echo $msg creating odir && mkdir -p $odir
   cd $odir  
}
epjconf-opdf(){ echo $(epjconf-odir)/$(epjconf-filename).pdf ; }
epjconf-open(){ open $(epjconf-opdf) ; }


#epjconf-confname(){ echo chep2018 ; }
epjconf-confname(){ echo chep2019 ; }

#epjconf-filename(){ echo opticks-blyth-$(epjconf-confname) ; }  # as submitted to referees
#epjconf-filename(){ echo opticks-blyth-$(epjconf-confname)-v1 ; }


#epjconf-filename(){ echo opticks-blyth-$(epjconf-confname)-v0 ; }
#epjconf-filename(){ echo opticks-blyth-$(epjconf-confname)-v1 ; }

epjconf-filename(){ echo opticks-snowmass21-loi-v0 ; }


epjconf-filename-notes(){ cat << EON

Bump the version, eg when accomodating referee comments::

    epsilon:chep2019 blyth$ epjconf-cd
    epsilon:chep2019 blyth$ pwd
    /Users/blyth/env/doc/epjconf/chep2019
    epsilon:chep2019 blyth$ l
    total 64
    -rw-r--r--  1 blyth  staff  29985 May 15 20:31 opticks-blyth-chep2019-v0.tex
    epsilon:chep2019 blyth$ 
    epsilon:chep2019 blyth$ cp opticks-blyth-chep2019-v0.tex opticks-blyth-chep2019-v1.tex
    epsilon:chep2019 blyth$ 

EON
}




epjconf-texname(){  echo $(epjconf-filename).tex ; }
epjconf-bibname(){  echo $(epjconf-filename) ; }

epjconf-etex(){     echo $(epjconf-dir)/$(epjconf-filename).tex ; }
epjconf-ebib(){     echo $(epjconf-dir)/opticks.bib ; }
epjconf-textmpl(){  echo template.tex ; }

epjconf-2016(){ open http://iopscience.iop.org/article/10.1088/1742-6596/898/4/042001/meta ; }

epjconf-info(){ cat << EOI

epjconf-tdir     : $(epjconf-tdir)
epjconf-textmpl  : $(epjconf-textmpl)
epjconf-dir      : $(epjconf-dir)
epjconf-texname  : $(epjconf-texname)
epjconf-filename : $(epjconf-filename)
epjconf-odir     : $(epjconf-odir) 
epjconf-opdf     : $(epjconf-opdf) 

epjconf-etex     : $(epjconf-etex) 
epjconf-ebib     : $(epjconf-ebib) 

EOI
}

epjconf-init-notes(){ cat << EON

$FUNCNAME
===================
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


epjconf-figdir(){ echo $HOME/simoncblyth.bitbucket.io ; }
epjconf-texinputs(){ echo .:$(epjconf-tdir):$(epjconf-figdir): ; }
epjconf-local-texinputs(){ echo .:$(epjconf-figdir): ; }

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
    epjconf-pdflatex
    epjconf-open
    epjconf-info
    ls -l $(epjconf-odir)    
}




epjconf-local-prep()
{
    local msg="=== $FUNCNAME "
    echo $msg
    epjconf-ocd

    local tdir=$(epjconf-tdir)
    cp $tdir/* .

    cp $(epjconf-etex) .
    cp $(epjconf-ebib) $(epjconf-bibname).bib
}

epjconf-local-pdflatex()
{
    local msg="=== $FUNCNAME "
    echo $msg
    epjconf-ocd
    TEXINPUTS=$(epjconf-local-texinputs) pdflatex $(epjconf-texname)
}
epjconf-local-bibtex()
{
    local msg="=== $FUNCNAME "
    echo $msg
    epjconf-ocd

    bibtex $(epjconf-bibname)
}
epjconf-local-clean()
{
    local msg="=== $FUNCNAME "
    echo $msg
    local odir=$(epjconf-odir)
    rm -rf $odir
}


epjconf--aborted-attempt-to-use-bibtex()
{
    epjconf-local-clean
    epjconf-local-prep
  
    epjconf-local-pdflatex
    epjconf-local-bibtex
    epjconf-local-pdflatex
    epjconf-local-pdflatex

    epjconf-open
    epjconf-info
    ls -l $(epjconf-odir)    
}



epjconf---notes(){ cat << EON

Cannot get bibtex to work in a clean source dir style, so 
copy everything into a tmpdir and run there

EON
}



epjconf-edit(){ local etex=$(epjconf-etex) ; local ebib=$(epjconf-ebib) ; echo $FUNCNAME etex $etex ebib $ebib  ; vi $etex $ebib $(epjconf-aux) ;  }
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

epjconf-ref(){ cp $(epjconf-opdf) ~/opticks_refs/ ; }
epjconf-oref(){ open ~/opticks_refs/$(epjconf-filename).pdf ; }

epjconf-lsref(){  ls -l ~/opticks_refs/*$(epjconf-confname)*.pdf ; }
