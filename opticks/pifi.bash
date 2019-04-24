pifi-source(){   echo ${BASH_SOURCE} ; }
pifi-vi(){       vi $(pifi-source) ; }
pifi-env(){      elocal- ; }
pifi-usage(){ cat << EOU

PIFI plans, reports
=====================

See also
---------

::

   reps-
   reps-cd


Number of English words to Chinese characters
-----------------------------------------------

* https://www.actranslation.com/chinese/chinese-wordcount.htm

::

    1000 Chinese characters ~ 600-700 English words
    1000 English words ~ 1500-1700 Chinese characters

    800 Chinese characters ~ 480-560 English words  => aim for 500 words



April 2019 Essay : Below is text copied from Word template 
---------------------------------------------------------------

::

    Essay Title    : Example: Working at CAS : An exciting and rewarding experience
    Name
    CAS Institute
    PIFI Category 


Contents
~~~~~~~~~

Please refer to the following aspects to write the contents, 
focusing on one or more that you would most like to describe.

(1) What is your feeling from having obtained the CAS-PIFI?
(2) What was your experience working at/with CAS? 
(3) What was your experience of living in China?
(4) **What is the most productive, impressive and rewarding work you have done with CAS?**
(5) What are your views on and suggestions for the development of CAS?
(6) What are your expectations for future collaboration with CAS?

Please attach at least two high-quality photos of your life and work at CAS.


Guidance
~~~~~~~~~

所有结题的项目（除中期验收项目），外国专家均须写一份500字词左右的在华工作感想（附件2）并提供三张工作照（图片格式），通过“结题核销”模块以附件形式分别上传，主要分享在华工作感受和生活故事，非学术性结题报告。未提交的项目视为结题验收不通过。

For all completed projects (except for mid-term acceptance projects), foreign
experts are required to write a 500-word work impression in China (Attachment
2) and provide three work photos (picture format), through “final verification”
The modules are uploaded separately as attachments, mainly sharing work
experience and life stories in China, non-academic final report. Unsubmitted
items are considered as acceptance of the completion of the project.



EOU
}
pifi-dir(){ echo $(dirname $(pifi-source)) ;  }
pifi-cd(){  cd $(pifi-dir); }
pifi-c(){   cd $(pifi-dir); }
pifi-wc(){ pifi-cd ; wc -w *.rst ; }

pifi-current(){ echo $(pifi-dir)/pifi_essay_april2019.rst  ; }
pifi-others(){  cat << EOO
$(pifi-dir)/pifi_progress_report_aug2018.rst 
EOO
}
pifi-e-(){ 
   pifi-current
   pifi-others
}
pifi-e(){ vi $($FUNCNAME-) ; }


pifi-tmp(){ echo /tmp/$USER/env/pifi ; }
pifi-tcd(){ cd $(pifi-tmp) ; }

pifi-rst2docx(){ 
   local msg="=== $FUNCNAME :"
   local path=$(pifi-current)
   local name=$(basename $path)
   local stem=${name/.rst}

   local tmp=$(pifi-tmp)
   mkdir -p $tmp
   local docx=$tmp/$stem.docx

   if [ -f "$docx" ]; then
       echo $msg docx exists already $docx
       local ans="NO"
       read -p "$msg enter YES to delete it and proceed, or anything else to abort " ans
       [ "$ans" == "YES" ] && echo $msg deleting docx $docx && rm $docx 
       [ "$ans" != "YES" ] && echo $msg ABORTing && return 
   fi


   local cmd="rst2docx.py $path $docx"
   echo $msg proceeding with $cmd
   eval $cmd

   ls -l $path $docx


   [ -f "$docx" ] && open $docx
}




pifi-words(){ echo hello world| wc -w ; }
