# === func-gen- : presentation/gtc2016/gtc fgp presentation/gtc2016/gtc.bash fgn gtc fgh presentation/gtc2016
gtc-src(){      echo presentation/gtc2016/gtc.bash ; }
gtc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gtc-src)} ; }
gtc-vi(){       vi $(gtc-source) ; }
gtc-env(){      elocal- ; }
gtc-usage(){ cat << EOU



Image approach::


    simon:gtc2016 blyth$ osx-
    simon:gtc2016 blyth$ osx-ss-cp sk-PH20-water-withboat-apr23-wm
    === osx-ss-cp : iwd /Users/blyth/env/presentation/gtc2016 rel presentation/gtc2016 repo env dir /Users/blyth/simoncblyth.bitbucket.org/env/presentation/gtc2016 dst /Users/blyth/simoncblyth.bitbucket.org/env/presentation/gtc2016/sk-PH20-water-withboat-apr23-wm.png
    total 15016
    -rw-r--r--@ 1 blyth  staff  6306459 Mar 15 13:27 dyb-pmt-wall-photo.png
    -rw-r--r--  1 blyth  staff  1377987 Mar 15 13:30 dyb-pmt-wall-photo_half.png
    cp "/Users/blyth/Desktop/Screen Shot 2016-03-15 at 1.46.43 PM.png" /Users/blyth/simoncblyth.bitbucket.org/env/presentation/gtc2016/sk-PH20-water-withboat-apr23-wm.png
    -rw-r--r--@ 1 blyth  staff  7942700 Mar 15 13:47 /Users/blyth/simoncblyth.bitbucket.org/env/presentation/gtc2016/sk-PH20-water-withboat-apr23-wm.png

    .. image:: /env/presentation/gtc2016/sk-PH20-water-withboat-apr23-wm.png
       :width: 900px
       :align: center

    simon:gtc2016 blyth$ 
    simon:gtc2016 blyth$ 
    simon:gtc2016 blyth$ downsize.py /Users/blyth/simoncblyth.bitbucket.org/env/presentation/gtc2016/sk-PH20-water-withboat-apr23-wm.png
    INFO:env.doc.downsize:Resize 2  
    INFO:env.doc.downsize:downsize /Users/blyth/simoncblyth.bitbucket.org/env/presentation/gtc2016/sk-PH20-water-withboat-apr23-wm.png to create /Users/blyth/simoncblyth.bitbucket.org/env/presentation/gtc2016/sk-PH20-water-withboat-apr23-wm_half.png 2362px_1542px -> 1181px_771px 
    simon:gtc2016 blyth$ 



Using fullsize PNG retina screen capture images but with half pixel html sizing, as
obtained by downsize.py.  This also creates half sized images named "_half.png".




EOU
}

gtc-sdir(){ echo $(env-home)/presentation/gtc2016 ; }
gtc-dir(){ echo $(local-base)/env/presentation/gtc2016 ; }
gtc-cd(){  
   local dir=$(gtc-dir)
   mkdir -p $dir
   cd $dir; 
}
gtc-scd(){  cd $(gtc-sdir); }

gtc-get()
{
    gtc-cd

    local url=http://www-sk.icrr.u-tokyo.ac.jp/sk/gallery/wme/PH20-water-withboat-apr23-wm.jpg

    [ ! -f "$(basename $url)" ] && curl -L -O $url 
}


