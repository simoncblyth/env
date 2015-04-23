# === func-gen- : tools/pages/pages fgp tools/pages/pages.bash fgn pages fgh tools/pages
pages-src(){      echo tools/pages/pages.bash ; }
pages-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pages-src)} ; }
pages-vi(){       vi $(pages-source) ; }
pages-env(){      elocal- ; }
pages-usage(){ cat << EOU


Manual approach to .rst to .doc conversion
--------------------------------------------

The attached .doc including figures was created 
by manually copying the text from the .rst into 
Word 2004 for Mac using "View > Outline" 
to manage the section hierarchy.

The table of content was created with 
"Insert > Index and tables"

On checking the document in Pages 5.2  I noticed
that the page breaks move around slightly.


With the Word 2004 example doc in pages
----------------------------------------

* Insert > TOC > Document    

  * succeeded to duplicate the TOC, but it took a few seconds
    during which it said "Generating TOC", 

  * perhaps there are no sections (or they didnt convert) 
    just heading styles ? 


Pages TOC
----------

* http://www.techradar.com/news/software/applications/10-top-pages-tips-and-tricks-980721

  * make sure you use defined paragraph styles for your headings





Can applescript be used to automate this ?
--------------------------------------------


* https://iworkautomation.com/pages/index.html

::

    tell application "Pages"
        --activate
        properties of front document 
        pages of front document 
        sections of front document   # only 1  for example

        set bt to body text of front document
        count of paragraphs of bt

        paragraph 12 of bt      # "Introduction" all prior to 12 are blanks
    end tell


::

    tell application "Pages"
        activate
        tell the front document
            tell body text
                font of character 1 
            end tell
        end tell
    end tell



https://iworkautomation.com/pages/body-text.html

https://iworkautomation.com/pages/body-text-basics.html

EOU
}
pages-dir(){ echo $(local-base)/env/tools/pages/tools/pages-pages ; }
pages-cd(){  cd $(pages-dir); }
pages-mate(){ mate $(pages-dir) ; }
pages-get(){
   local dir=$(dirname $(pages-dir)) &&  mkdir -p $dir && cd $dir

}
