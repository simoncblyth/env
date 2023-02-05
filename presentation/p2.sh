#!/bin/bash -l 

usage(){ cat << EOU
p2.sh : presentation with annotation s5_talk pages interleaved
=================================================================

Running p2.sh opens two tabs in Safari.app:

1. normal presentation pages
2. normal presentation pages with s5_talk annotations interleaved

Use p2.sh whilst thinking about what you are going to say 
in a presentation and updating the s5_talk blocks 
following every presentation page. 

See also:

* slides- ; slides-vi 
* presentation- ; presentation-vi
* env/presentation/s5_talk.py 


Handling Lots of Annotation
----------------------------

If there is too much text to fit on one page start the
text with "SMALL".  For examples::

    grep SMALL *.txt

Note that the sizing is implemented in rst2rst.py::

    epsilon:env blyth$ find . -name '*.py' -exec grep -H SMALL {} \;
    ./bin/rst2rst.py:                if lines[i+2].find("SMALL") > -1:

EOU
}

presentation-
iname=$(presentation-iname)

INAME=${iname} presentation--
INAME=${iname}_TALK presentation--


