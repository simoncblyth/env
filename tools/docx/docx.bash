# === func-gen- : tools/docx/docx fgp tools/docx/docx.bash fgn docx fgh tools/docx
docx-src(){      echo tools/docx/docx.bash ; }
docx-source(){   echo ${BASH_SOURCE:-$(env-home)/$(docx-src)} ; }
docx-vi(){       vi $(docx-source) ; }
docx-env(){      elocal- ; }
docx-usage(){ cat << EOU

Python-docx
=============

python-docx is a Python library for creating 
and updating Microsoft Word (.docx) files.

* https://python-docx.readthedocs.org/en/latest/
* https://python-docx.readthedocs.io/en/latest/user/text.html
* https://python-docx.readthedocs.org/en/latest/user/styles-understanding.html

Hmm looks like macports drastically out of date
-------------------------------------------------

* https://github.com/mikemaccana/python-docx
* https://github.com/python-openxml/python-docx



::

    epsilon:docutils_ blyth$ port info py27-docx
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    py27-docx @0.0.2_1 (python, devel)
    Variants:             universal

    Description:          The module was created when I was looking for a Python support for MS Word .doc files, but could only find various hacks involving COM automation,
                          calling .net or Java, or automating OpenOffice or MS Office.
    Homepage:             https://github.com/mikemaccana/python-docx

    Fetch Dependencies:   git
    Library Dependencies: python27, py27-lxml
    Platforms:            darwin
    License:              MIT
    Maintainers:          none
    epsilon:docutils_ blyth$ 



Installing
------------

* https://python-docx.readthedocs.io/en/latest/user/install.html

1. installed py27-pip using macports

2. then : sudo pip install python-docx

::

    epsilon:presentation blyth$ sudo pip install python-docx
    The directory '/Users/blyth/Library/Caches/pip/http' or its parent directory is not owned by the current user and the cache has been disabled. Please check the permissions and owner of that directory. If executing pip with sudo, you may want sudo's -H flag.
    The directory '/Users/blyth/Library/Caches/pip' or its parent directory is not owned by the current user and caching wheels has been disabled. check the permissions and owner of that directory. If executing pip with sudo, you may want sudo's -H flag.
    Collecting python-docx
      Downloading https://files.pythonhosted.org/packages/00/ed/dc8d859eb32980ccf0e5a9b1ab3311415baf55de208777d85826a7fb0b65/python-docx-0.8.7.tar.gz (5.4MB)
        100% |████████████████████████████████| 5.4MB 12kB/s 
    Requirement already satisfied: lxml>=2.3.2 in /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages (from python-docx) (4.1.1)
    Installing collected packages: python-docx
      Running setup.py install for python-docx ... done
    Successfully installed python-docx-0.8.7
    epsilon:presentation blyth$ 

::

     -H, --set-home
                   Request that the security policy set the HOME environment variable to the home directory specified by the target user's password
                   database entry.  Depending on the policy, this may be the default behavior.

::

    epsilon:opticks blyth$ sudo -H echo $(eval echo \$HOME)
    /Users/blyth
    epsilon:opticks blyth$ sudo echo $(eval echo \$HOME)
    /Users/blyth



Definition list support ?
---------------------------

Workaround by using bullet list instead.

::

    2016-05-08 09:43:52,691 env.doc.docutils.rst2docx INFO     reading ntu-report-may-2016.rst 
    NotImplementedError: env.doc.docutils.rst2docx.Translator visiting unknown node type: definition_list

Emphasis and Strong support 
------------------------------

Added by reworking the translation to docx.



See also
---------

* docxbuilder-


EOU
}
docx-dir(){  echo $(local-base)/env/tools/docx ; }
docx-sdir(){ echo $(env-home)/tools/docx ; }
docx-scd(){  cd $(docx-sdir); }
docx-cd(){   cd $(docx-dir); }

docx-get()
{
   local dir=$(dirname $(docx-dir)) &&  mkdir -p $dir && cd $dir

    git clone https://github.com/python-openxml/python-docx docx
}

docx-ln()
{
    python-
    python-ln $(docx-dir)/docx
}

docx-test()
{
    cd ~/workflow/admin/reps

    rst2docx.py test.rst /tmp/test.docx

    open /tmp/test.docx
}

