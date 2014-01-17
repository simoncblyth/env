ROOT issues
============

freetype ftgl issue with v5.34.14 (2014/01/17)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     6860 
     6861     In file included from /usr/local/env/chroma_env/src/root-v5.34.14/graf2d/graf/src/TMathText.cxx:15:
     6862 
     6863     include/TTF.h:51:4: error: unknown type name 'FT_Glyph'; did you mean 'FTGlyph'?
     6864 
     6865        FT_Glyph   fImage; // glyph image
     6866 
     6867        ^~~~~~~~
     6868 
     6869        FTGlyph
     6870 
     6871     include/ftglyph.h:25:19: note: 'FTGlyph' declared here
     6872 
     6873     class FTGL_EXPORT FTGlyph
     6874 

* http://trac.macports.org/ticket/41572
* https://github.com/root-mirror/root/commit/446a11828dcf577efd15d9057703c5bd099dd148
* https://sft.its.cern.ch/jira/browse/ROOT-5773
* http://root.cern.ch/phpBB3/viewtopic.php?f=3&t=17485


Comparing git master with v5.34.14 shows more diffs than needed.::

    (chroma_env)delta:root-v5.34.14 blyth$ diff /tmp/r/root/graf3d/ftgl/Module.mk /usr/local/env/chroma_env/src/root-v5.34.14/graf3d/ftgl/Module.mk
    25c25
    < # ALLHDRS     +=
    ---
    > ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(FTGLH))
    31,32d30
    < FTGLINC       := -I$(MODDIRI)
    < 
    64c62
    < $(FTGLO):     CXXFLAGS += $(FREETYPEINC) $(FTGLINC) $(OPENGLINCDIR:%=-I%)
    ---
    > $(FTGLO):     CXXFLAGS += $(FREETYPEINC) $(OPENGLINCDIR:%=-I%)
    (chroma_env)delta:root-v5.34.14 blyth$ 


Kludge patch application as v5.34.15 is not yet released::

    (chroma_env)delta:root blyth$ git diff --patch 0499d19a7d21de7dd4b895e8540460b389a238f4  446a11828dcf577efd15d9057703c5bd099dd148 > /usr/local/env/chroma_env/src/root-v5.34.14.patch01file
    (chroma_env)delta:root blyth$ pwd
    /tmp/r/root


::

    (chroma_env)delta:src blyth$ cp -r root-v5.34.14 root-v5.34.14.patch01
    (chroma_env)delta:src blyth$ cd root-v5.34.14.patch01
    (chroma_env)delta:root-v5.34.14.patch01 blyth$ patch -p1 < ../root-v5.34.14.patch01file
    patching file graf3d/ftgl/Module.mk
    patching file graf3d/gl/Module.mk
    (chroma_env)delta:root-v5.34.14.patch01 blyth$ 
    (chroma_env)delta:root-v5.34.14.patch01 blyth$ cd ..
    (chroma_env)delta:src blyth$ diff -r root-v5.34.14 root-v5.34.14.patch01
    diff -r root-v5.34.14/graf3d/ftgl/Module.mk root-v5.34.14.patch01/graf3d/ftgl/Module.mk
    25c25
    < ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(FTGLH))
    ---
    > # ALLHDRS     +=
    30a31,32
    > FTGLINC       := -I$(MODDIRI)
    > 
    62c64
    < $(FTGLO):     CXXFLAGS += $(FREETYPEINC) $(OPENGLINCDIR:%=-I%)
    ---
    > $(FTGLO):     CXXFLAGS += $(FREETYPEINC) $(FTGLINC) $(OPENGLINCDIR:%=-I%)
    diff -r root-v5.34.14/graf3d/gl/Module.mk root-v5.34.14.patch01/graf3d/gl/Module.mk
    66a67,68
    > FTGLINC       := -I$(MODDIRI)/../../ftgl/inc
    > 
    120c122
    < $(GLO): CXXFLAGS += $(GLEWINCDIR:%=-I%) $(GLEWCPPFLAGS)
    ---
    > $(GLO): CXXFLAGS += $(GLEWINCDIR:%=-I%) $(GLEWCPPFLAGS) $(FTGLINC)
    (chroma_env)delta:src blyth$ 
    (chroma_env)delta:src blyth$ pwd
    /usr/local/env/chroma_env/src















