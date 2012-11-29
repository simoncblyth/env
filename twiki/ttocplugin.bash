# === func-gen- : twiki/ttocplugin fgp twiki/ttocplugin.bash fgn ttocplugin fgh twiki
ttocplugin-src(){      echo twiki/ttocplugin.bash ; }
ttocplugin-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ttocplugin-src)} ; }
ttocplugin-vi(){       vi $(ttocplugin-source) ; }
ttocplugin-env(){      elocal- ; }
ttocplugin-usage(){ cat << EOU

TocPlugin for TWiki
======================


Installation 
-------------

::

	sh-3.2# pwd 
	/usr/local/env/TWiki-5
	sh-3.2# curl -L -O http://twiki.org/p/pub/Plugins/TocPlugin/TocPlugin.zip
	  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
					 Dload  Upload   Total   Spent    Left  Speed
	100 31649  100 31649    0     0  56438      0 --:--:-- --:--:-- --:--:-- 98288
	sh-3.2# unzip -l TocPlugin.zip 
	Archive:  TocPlugin.zip
	  Length     Date   Time    Name
	 --------    ----   ----    ----
		0  12-12-08 18:26   data/
		0  12-12-08 18:26   data/TWiki/
	       42  12-12-08 18:26   data/TWiki/WebOrder.txt
	     3534  12-12-08 18:26   data/TWiki/TocPlugin.txt
	    10735  12-12-08 18:26   data/TWiki/TocPluginHelp.txt
		0  12-12-08 18:26   lib/
		0  12-12-08 18:26   lib/TWiki/
		0  12-12-08 18:26   lib/TWiki/Plugins/
	     3052  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin.pm
		0  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/
		0  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/test/
	     4327  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/test/TopLevelSectionTests.pl
		0  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/test/TWiki/
	       45  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/test/TWiki/Func.pm
	      823  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/test/AttrsTests.pl
	     9839  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/test/SectionTests.pl
	     1828  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/test/Assert.pm
	      376  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/test/HTML.pm
	     1982  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/test/TOCTests.pl
	     1159  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/test/FakeWikiIF.pm
	     1353  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/test/AnchorTests.pl
	     2211  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/TOCIF.pm
	    17971  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/Section.pm
	     1715  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/Attrs.pm
	     5839  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/TOC.pm
	     3923  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/Anchor.pm
	     6278  12-12-08 18:26   lib/TWiki/Plugins/TocPlugin/TopLevelSection.pm
	     4285  12-12-08 18:26   TocPlugin_installer
	 --------                   -------
	    81317                   28 files
	sh-3.2# 


	sh-3.2# unzip TocPlugin.zip 
	Archive:  TocPlugin.zip
	  inflating: data/TWiki/WebOrder.txt  
	  inflating: data/TWiki/TocPlugin.txt  
	  inflating: data/TWiki/TocPluginHelp.txt  
	  inflating: lib/TWiki/Plugins/TocPlugin.pm  
	   creating: lib/TWiki/Plugins/TocPlugin/
	   creating: lib/TWiki/Plugins/TocPlugin/test/
	  inflating: lib/TWiki/Plugins/TocPlugin/test/TopLevelSectionTests.pl  
	   creating: lib/TWiki/Plugins/TocPlugin/test/TWiki/
	 extracting: lib/TWiki/Plugins/TocPlugin/test/TWiki/Func.pm  
	  inflating: lib/TWiki/Plugins/TocPlugin/test/AttrsTests.pl  
	  inflating: lib/TWiki/Plugins/TocPlugin/test/SectionTests.pl  
	  inflating: lib/TWiki/Plugins/TocPlugin/test/Assert.pm  
	  inflating: lib/TWiki/Plugins/TocPlugin/test/HTML.pm  
	  inflating: lib/TWiki/Plugins/TocPlugin/test/TOCTests.pl  
	  inflating: lib/TWiki/Plugins/TocPlugin/test/FakeWikiIF.pm  
	  inflating: lib/TWiki/Plugins/TocPlugin/test/AnchorTests.pl  
	  inflating: lib/TWiki/Plugins/TocPlugin/TOCIF.pm  
	  inflating: lib/TWiki/Plugins/TocPlugin/Section.pm  
	  inflating: lib/TWiki/Plugins/TocPlugin/Attrs.pm  
	  inflating: lib/TWiki/Plugins/TocPlugin/TOC.pm  
	  inflating: lib/TWiki/Plugins/TocPlugin/Anchor.pm  
	  inflating: lib/TWiki/Plugins/TocPlugin/TopLevelSection.pm  
	  inflating: TocPlugin_installer     
	sh-3.2# 


Config TWiki
--------------

Enabled from http://localhost/twiki/bin/configure

#. enter configure password
#. choose "Open All Options" then a Plugins link should appear 
#. check/uncheck the checkbox for TocPlugin
#. click [Next] then [Save]

#. when enabling get::

    $TWiki::cfg{Plugins}{TocPlugin}{Enabled} = 1;



Registering a new user requires to be able to send email
---------------------------------------------------------

Hmm seems not, can login without getting the link in the mail 



EOU
}
ttocplugin-dir(){ echo $(local-base)/env/twiki/twiki-ttocplugin ; }
ttocplugin-cd(){  cd $(ttocplugin-dir); }
ttocplugin-mate(){ mate $(ttocplugin-dir) ; }
ttocplugin-get(){
   local dir=$(dirname $(ttocplugin-dir)) &&  mkdir -p $dir && cd $dir

}


