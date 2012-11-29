# === func-gen- : twiki/twiki fgp twiki/twiki.bash fgn twiki fgh twiki
twiki-src(){      echo twiki/twiki.bash ; }
twiki-source(){   echo ${BASH_SOURCE:-$(env-home)/$(twiki-src)} ; }
twiki-vi(){       vi $(twiki-source) ; }
twiki-env(){      elocal- ; }
twiki-usage(){ cat << EOU

TWIKI
======


Download
---------
Annoyingly SF does not deign to give a direct link, have 
to fill in contact form to get to download

::

    mv ~/Downloads/TWiki-5/ /usr/local/env/


Basics
-------

::

    sudo  chown -R www:www /usr/local/env/TWiki-5/
    cd /Library/WebServer/Documents
    ln -s /usr/local/env/TWiki-5 twiki
    open http://localhost/twiki/ 

As root, fixup perl paths to the macports one::

    twiki-
    twiki-cd bin
    sudo su


    perl -pi -e 's,/usr/bin/perl,/opt/local/bin/perl,g' *
    cp LocalLib.cfg.txt LocalLib.cfg

    chown www:www  LocalLib.cfg 

    Edit the cfg setting: $twikiLibPath = "/usr/local/env/TWiki-5/lib";

    ## from the root
    cat twiki_httpd_conf.txt | perl -p -e 's,/home/httpd/,/Library/WebServer/Documents/,g' - > twiki.conf

    ## apache-edit to add the below lines to apache httpd.conf

        # symbolic link to /usr/local/env/TWiki-5
        Include /Library/WebServer/Documents/twiki/twiki.conf

    open http://localhost/twiki/bin/configure  # choose the settings page and save

    Now have a twiki

    http://localhost/twiki/bin/view/ 

    Could not perform search. Error was: exec failed: No such file or directory /bin/grep   ## its 

    sh-3.2# ln -s /usr/bin/grep grep


    http://twiki.org/cgi-bin/view/Plugins/WebHome

    Hmm the TocPlugin is ancient http://twiki.org/cgi-bin/view/Plugins/TocPlugin










Install
--------

 * http://twiki.org/cgi-bin/view/TWiki/TWikiInstallationGuide
 * http://twiki.org/cgi-bin/view/Support/WebHome





EOU
}
twiki-dir(){ echo $(local-base)/env/TWiki-5 ; }
twiki-cd(){  cd $(twiki-dir)/$1; }
twiki-mate(){ mate $(twiki-dir) ; }
twiki-get(){
   local dir=$(dirname $(twiki-dir)) &&  mkdir -p $dir && cd $dir

}
