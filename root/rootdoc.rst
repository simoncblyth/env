
ROOT Documentation
===================


::

	cd $ROOTSYS
	make html



Build Problems
---------------

Does not build cleanly. Get multiple segv and hangs, but keep repeating and gets further each time.


::

       1594 htmldoc/TArrow.html
       1593 htmldoc/TArrowEditor.html

       *** Break *** segmentation violation


::

        974 htmldoc/TGTable.html
        Error in <TFile::TFile>: file /data1/env/local/dyb/external/ROOT/5.26.00e_python2.7/i686-slc5-gcc41-dbg/root/tutorials/hsimple.root does not exist
        Error in <TTreeTableInterface>: No tree supplied

         *** Break *** segmentation violation




