NuWa on Mavericks
===================


Getting NuWa installed on Mavericks, with clang will take effort::

    delta:~ blyth$ dybinst-
    delta:~ blyth$ dybinst-cd
    delta:dyb blyth$ dybinst-all
    Setting MACOSX_DEPLOYMENT_TARGET=10.9
    Checking out installation directory installation/trunk/dybinst.
    Making directory ./external


    Wed Feb 12 11:19:47 CST 2014
    Start Logging to /usr/local/env/dyb/dybinst-20140212-111947.log (or dybinst-recent.log)


    Starting dybinst commands: cmt checkout external optional projects

    Stage: "cmt"... 

    Downloading http://dayabay.bnl.gov/software/offline/tarFiles/CMT/v1r20p20080222/CMTv1r20p20080222.tar.gz ...done
    Unpacking CMTv1r20p20080222.tar.gz ...done
    Building CMT ...
    Stage: "checkout"... 

    SVN checkout http://dayabay.ihep.ac.cn/svn/dybsvn/NuWa/trunk to /usr/local/env/dyb/NuWa-trunk ...
    done
    Command: svn co -q http://dayabay.ihep.ac.cn/svn/dybsvn/NuWa/trunk /usr/local/env/dyb/NuWa-trunk
    done
    Found CMTCONFIG="i386-darwin-UnknownCompiler-dbg" from lcgcmt
    Checking your CMTCONFIG="i386-darwin-UnknownCompiler-dbg"...
    #CMT> Warning: apply_tag with empty name [$(cmt_compiler_version)]
    #CMT> Warning: apply_tag with empty name [$(cmt_compiler_version)]

    ERROR: CMTCONFIG is not compatible!

    CMTCONFIG is:   "i386-darwin-UnknownCompiler-dbg" 
    Should be like: "i386-darwin-UnknownCompiler-dbg"

    You may be explicitly defining CMTCONFIG or you are using a platform
    that is not yet supported.  If you don't care about what CMTCONFIG is
    set to, then do not set it prior to running dybinst.  If you do want
    to force CMTCONFIG or you have a new platform you must add support

    To add support you must edit:

      NuWa-trunk/lcgcmt/LCG_Settings/cmt/requirements

    1) If the CMTCONFIG shown above has any "Unknown" parts you must add
    various host-* tags based on the intrisic output of:

    $ cmt show tags
    CMTv1 (from CMTVERSION)
    CMTr20 (from CMTVERSION)
    CMTp20080222 (from CMTVERSION)
    Darwin (from uname) package CMT implies [Unix]
    i386 (from package CMT)
    mac109 (from package CMT)
    Unix (from package CMT) excludes [WIN32 Win32]

    And then you may need to add to the "host-os", "host-compiler" and
    "host-cpu" macros definitions.

    2) To tell CMT what your platform characteristcs are, you must define
    various "tag-*" tags based on the CMTCONFIG tag.  The standard form is

    ### Daya Bay CMTCONFIG tags
    ...
    tag i386-darwin-UnknownCompiler-dbg tag-XXX tag-YYY tag-ZZZ tag-dbg
    tag i386-darwin-UnknownCompiler-opt tag-XXX tag-YYY tag-ZZZ tag-opt
    ### end Daya Bay CMTCONFIG tags

    Note, replace any "Unknown" with what you fixed in step 1.

     failed with 1



Bring new config to NuWa
--------------------------

Seemingly matching CMTCONFIG that still cause error:

* http://dayabay.ihep.ac.cn/tracs/dybsvn/ticket/257
* http://dayabay.ihep.ac.cn/tracs/dybsvn/changeset/6776

Quoting from the ticket::

   Even when the CMTCONFIG comes out matching ... can get a failure if your combination of
   targets is not in LCG_basesystem.

Looking at the revisions of others doing this can help

* http://dayabay.ihep.ac.cn/tracs/dybsvn/log/lcgcmt/trunk/LCG_Settings/cmt/requirements


