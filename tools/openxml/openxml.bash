# === func-gen- : tools/openxml/openxml fgp tools/openxml/openxml.bash fgn openxml fgh tools/openxml
openxml-src(){      echo tools/openxml/openxml.bash ; }
openxml-source(){   echo ${BASH_SOURCE:-$(env-home)/$(openxml-src)} ; }
openxml-vi(){       vi $(openxml-source) ; }
openxml-env(){      elocal- ; }
openxml-usage(){ cat << EOU

OpenXML : code access to word/... docs
========================================

https://github.com/OfficeDev/Open-XML-SDK

The Open XML SDK provides open-source libraries for working with Open XML 
Documents (DOCX, XLSX, and PPTX).  

It supports scenarios such as: 

- High-performance generation of word-processing documents, spreadsheets,
  and presentations
- Document modification, such as removing tracked revisions or removing
  unacceptable content from documents
- Data and content querying and extraction, such as transformation from
  DOCX to HTML, or extraction of data from spreadsheets

Open XML SDK 2.5 for Office

* http://msdn.microsoft.com/en-us/library/office/bb448854.aspx

How to..

* https://msdn.microsoft.com/en-us/library/office/cc850833.aspx

Refs
----

* http://ericwhite.com/blog/openxml-expanded/




Intro Screen Cast
---------------------

As in the screen cast

* http://openxmldeveloper.org/blog/b/openxmldeveloper/archive/2014/07/03/screen-cast-using-open-xml-sdk-on-linux-using-mono.aspx
* https://github.com/OfficeDev/Open-XML-SDK/pull/3
* http://stackoverflow.com/questions/25875908/mono-binary-cant-find-dlls-in-subdirectory

The below functions are a way to avoid copying dll around using a 
config file (supposedly absolute paths should not work). But
seems to be working.



* http://www.codeitive.com/0NNqgPWkVX/openxml-cannot-open-package-because-filemode-or-fileaccess-value-is-not-valid-for-the-stream.html

Had to use isEditable true 

::

    delta:openxml blyth$ openxml-;openxml-run extractstyles.cs /tmp/style.docx  ## succeeds to pull out xml



PowerShell
-----------

* http://blogs.msdn.com/b/ericwhite/archive/2008/06/11/processing-open-xml-documents-server-side-using-powershell.aspx
* 

Examples
-----------

* http://blogs.msdn.com/b/ericwhite/archive/2008/07/09/open-xml-sdk-and-linq-to-xml.aspx



EOU
}
openxml-dir(){    echo $(local-base)/env/tools/openxml/Open-XML-SDK ; }
openxml-bindir(){ echo $(local-base)/env/tools/openxml/bin ; }
openxml-cd(){  cd $(openxml-dir); }

openxml-sdir(){ echo $(env-home)/tools/openxml ; }
openxml-scd(){ cd $(openxml-sdir) ; }


openxml-mate(){ mate $(openxml-dir) ; }
openxml-get(){
   local dir=$(dirname $(openxml-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/OfficeDev/Open-XML-SDK
}

openxml-make(){
   openxml-cd
   make -f Makefile-Linux-Mono build
}

openxml-dlldir(){ echo $(openxml-dir)/build/OpenXmlSdkLib ; }

openxml-config-(){ cat << EOC

<?xml version="1.0" encoding="utf-8" ?>
<configuration>
  <runtime>
    <assemblyBinding xmlns="urn:schemas-microsoft-com:asm.v1">
      <probing privatePath="$(openxml-dlldir)" />
    </assemblyBinding>
  </runtime>
</configuration>

EOC

}

openxml-bin(){
   local path=${1:-helloxml.cs}
   local name=${path/.cs}
   local bin=$(openxml-bindir)/$name
   echo $bin
}

openxml-compile(){
   local path=${1:-helloxml.cs}
   local bin=$(openxml-bin $path)
   mkdir -p $(dirname $bin)
   local cmd="mcs -r:$(openxml-dlldir)/OpenXMLLib.dll -r:System.Xml.Linq.dll -r:WindowsBase.dll $path -out:$bin "
   echo $cmd
   eval $cmd

   openxml-config- 
  
   echo
   echo writing above config to $bin.config 
   openxml-config- > $bin.config
}


openxml-run(){
   local path=${1:-helloxml.cs}
   local bin=$(openxml-bin $path)
   shift 
   mono $bin $* 
}

openxml-helloxml(){
   openxml-scd
   openxml-run helloxml.cs helloxml.docx
}

