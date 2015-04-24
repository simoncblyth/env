# === func-gen- : tools/docxbuilder/docxbuilder fgp tools/docxbuilder/docxbuilder.bash fgn docxbuilder fgh tools/docxbuilder
docxbuilder-src(){      echo tools/docxbuilder/docxbuilder.bash ; }
docxbuilder-source(){   echo ${BASH_SOURCE:-$(env-home)/$(docxbuilder-src)} ; }
docxbuilder-vi(){       vi $(docxbuilder-source) ; }
docxbuilder-env(){      elocal- ; }
docxbuilder-usage(){ cat << EOU

Sphinx-docxbuilder
====================

From sphinx direct to docx : gotta try this

* https://bitbucket.org/haraisao/sphinx-docxbuilder

From README

    This programs is an extension to generate a docx file with Sphinx-1.1.2.  
    This extension is developed by hacking both 'sphinxcontrib-docxbuilder' and
    'python-docx'.

My macports Sphinx is 1.2 

Apparently docxbuilder incorporates parts of those other packages, it 
doesnt depend on them.


Issue : hyphenated extension name is invalid python module name ?
----------------------------------------------------------------------

Use "docxbuilder" rather than "sphinx-docxbuilder"

Oops hardcoded names::

    delta:docxbuilder blyth$ find . -name '*.py' -exec grep -H sphinx-docxbuilder {} \;

    ./__init__.py:    app.add_config_value('docx_creator', 'sphinx-docxbuilder', 'env')
    ./contrib/quickstart.py:extensions = ['sphinx-docxbuilder',%(extensions)s]
    ./docx/docx.py:    #fname = find_file(stylefile, 'sphinx-docxbuilder/docx')
    delta:docxbuilder blyth$ 



Issue : getting builder registered
-------------------------------------

::

    Sphinx error:
    Builder name docx not registered

* http://sphinx-doc.org/builders.html

Huh, registration seems to have resolved itself 
after adding logging to the docxbuilder and conf.py 


Issue : preparing documents... Error: style file( style.docx ) not found
---------------------------------------------------------------------------

::

  File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/docxbuilder/docx/docx.py", line 1681, in relationshiplist
    os.chdir(self.template_dir)

This was due to hardcoded extension names in the source.


Issue : source is littered with tabs, but not consistently
-------------------------------------------------------------

Issue  : assumption of document contents ?
--------------------------------------------

::

    2015-04-24 12:40:06,832 docxbuilder.writer INFO     visit_compound states:[[], []] 

    Exception occurred:
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/docxbuilder/writer.py", line 453, in visit_compound
        if self.states[-1][0]  == 'Contents:' :
    IndexError: list index out of range

Just get it to run::

     452         log.info("visit_compound states:%s " % repr(self.states))$
     453         if len(self.states[-1]) and self.states[-1][0]  == 'Contents:' :$
     454             self.states.pop()$
     455             self.states.append(['  '])$



Issue : funny styling
-----------------------

Succeed to make a .docx that Pages can open, but litany of issues


* cover page with docxbuilder author name on it 

* document title not propagated

* no table of contents, generating one with pages fails to 
  notice the two-level heirarchy of the original

* double spaced ragged text layout, making documnent twice the length 
  as with latexpdf builder

* silly ragged centered figure caption text 





Tried taking Pages template, exporting to docx and using that as style.docx
-----------------------------------------------------------------------------

Have 3 levels, they all appear with same styling in Pages generated TOC

* Heading: name
* Heading 2: major sections
* Heading 3: minor sections

* https://discussions.apple.com/thread/5508910

Just select one heading, and change style : changes for all in the toc 
(not working in the body)


* https://discussions.apple.com/thread/6184230

Sounds like you have not applied the Headings style to the other headings.
 
Select the headings one by one (because it is Pages 5.2 and brain dead) and
double click the heading style to force the style onto the headings. Once you
have done that you can update the heading style and it will apply to all the
headings styled with that style.


Looking into how docxbuilding and docx work
---------------------------------------------

* docxbuilder provides a visitor for passing over
  the doctrees nodes and emitting docx commands such
  as docx.heading

* docx is lxml element tree based, generating an xml document 


openxml toc
-------------

* http://openxmldeveloper.org/blog/b/openxmldeveloper/archive/2011/08/10/exploring-tables-of-contents-in-open-xml-wordprocessingml-documents-part-2.aspx

The example code shows how to insert a TOC at a desired point in a document,
and then set the <w:updateFields> element.  Then, when the user next opens that
document, Word will present them with the option to repaginate and update the
TOC.

As part of the definition of each TOC, you specify a set of switches that Word
uses as instructions on how to construct the TOC.  This screen-cast discusses
the TOC switches, and shows how to find out more about them from the text of
the Open XML standard.

* http://stackoverflow.com/questions/9762684/how-to-generate-table-of-contents-using-openxml-sdk-2-0



Observations
--------------

* Pages 5.2 is brain dead, fails to provide a simple way to change all header styles in one place

* Perhaps can use the docx template as the source of style info that gets repeated
  into the document ?  


docxbuilder writer init
-------------------------------

::

     docx_style = "test.docx"   ## conf.py setting 

     127     def __init__(self, builder):
     128         writers.Writer.__init__(self)
     129         self.builder = builder
     130         self.docx = docx.DocxComposer()
     131 
     132         self.title = self.builder.config['docx_title']
     133         self.subject = self.builder.config['docx_subject']
     ...
     144         stylefile = self.builder.config['docx_style']
     145         if stylefile :
     146             self.docx.new_document(stylefile)
     147         else:
     148             self.docx.new_document('style.docx')


DocxComposer::

     673   def new_document(self, stylefile):
     674     '''
     675        Preparing a new document
     676     '''
     677     log.info("new_document stylefile %s " % stylefile)
     678     self.set_style_file(stylefile)
     679     self.document = make_element_tree([['w:document'],[['w:body']]])
     680     self.docbody = get_elements(self.document, '/w:document/w:body')[0]
     681     self.current_docbody = self.docbody
     682 
     683     self.relationships = self.relationshiplist()
     684 
     685     return self.document

::

     610   def set_style_file(self, stylefile):
     614     #fname = find_file(stylefile, 'sphinx-docxbuilder/docx')
     615     fname = find_file(stylefile, 'docxbuilder/docx')
     ...
     621     self.styleDocx = DocxDocument(fname)
     622 
     623     self.template_dir = tempfile.mkdtemp(prefix='docx-')
     624     result = self.styleDocx.extract_files(self.template_dir)
     ...
     632     self.stylenames = self.styleDocx.extract_stylenames()
     633     self.paper_info = self.styleDocx.get_paper_info()
     634     self.bullet_list_indents = self.get_numbering_left('ListBullet')
     635     self.bullet_list_numId = self.styleDocx.get_numbering_style_id('ListBullet')
     636     self.number_list_indent = self.get_numbering_left('ListNumber')[0]
     637     self.number_list_numId = self.styleDocx.get_numbering_style_id('ListNumber')
     638     self.abstractNums = get_elements(self.styleDocx.numbering, 'w:abstractNum')
     639     self.numids = get_elements(self.styleDocx.numbering, 'w:num')
     640     self.numbering = make_element_tree(['w:numbering'])


::

     253 class DocxDocument:
     254   def __init__(self, docxfile=None):
     ... 
     267     if docxfile :
     268       self.set_document(docxfile)
     269       self.docxfile = docxfile
     270 
     271   def set_document(self, fname):
     ...  
     276       self.docxfile = fname
     277       self.docx = zipfile.ZipFile(fname)
     278 
     279       self.document = self.get_xmltree('word/document.xml')
     280       self.docbody = get_elements(self.document, '/w:document/w:body')[0]
     281 
     282       self.numbering = self.get_xmltree('word/numbering.xml')
     283       self.styles = self.get_xmltree('word/styles.xml')
     284       self.extract_stylenames()
     285       self.paragraph_style_id = self.stylenames['Normal']
     286       self.character_style_id = self.stylenames['Default Paragraph Font']
     287 
     288 
     289     return self.document




DocxComposer::

     699   def save(self, docxfilename):
     ...
     705     self.coreproperties()
     706     self.appproperties()
     707     self.contenttypes()
     708     self.websettings()
     709 
     710     self.wordrelationships()
     711 
     712     for x in self.abstractNums :
     713       self.numbering.append(x)
     714     for x in self.numids :
     715       self.numbering.append(x)
     716 
     717     coverpage = self.styleDocx.get_coverpage()
     718 
     719     if not self.nocoverpage and coverpage is not None :
     720       print "output Coverpage"
     721       self.docbody.insert(0,coverpage)
     722 
     723     self.docbody.append(self.paper_info)
     724 
     725 
     726     # Serialize our trees into out zip file
     727     treesandfiles = {self.document:'word/document.xml',
     728                      self._coreprops:'docProps/core.xml',
     729                      self._appprops:'docProps/app.xml',
     730                      self._contenttypes:'[Content_Types].xml',
     731                      self.numbering:'word/numbering.xml',
     732                      self.styleDocx.styles:'word/styles.xml',
     733                      self._websettings:'word/webSettings.xml',
     734                      self._wordrelationships:'word/_rels/document.xml.rels'}
     735 
     736     docxfile = self.styleDocx.restruct_docx(self.template_dir, docxfilename, treesandfiles.values())
     737 
     738     for tree in treesandfiles:
     739         if tree != None:
     740             #print 'Saving: '+treesandfiles[tree]    
     741             treestring =  etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone='yes')
     742             docxfile.writestr(treesandfiles[tree],treestring)
     743 
     744     print 'Saved new file to: '+docxfilename
     745     shutil.rmtree(self.template_dir)
     746     return


::

     388   def restruct_docx(self, docx_dir, docx_filename, files_to_skip=[]):
     389     '''
     390        This function is copied and modified the 'savedocx' function contained 'python-docx' library
     391       Restruct docx file from files in 'doxc_dir'
     392     '''
     ...
     397     docxfile = zipfile.ZipFile(docx_filename, mode='w', compression=zipfile.ZIP_DEFLATED)
     398 
     399     prev_dir = os.path.abspath('.')
     400     os.chdir(docx_dir)
     401 
     402     # Add & compress support files
     403     files_to_ignore = ['.DS_Store'] # nuisance from some os's
     404     for dirpath,dirnames,filenames in os.walk('.'):
     405         for filename in filenames:
     406             if filename in files_to_ignore:
     407                 continue
     408             templatefile = join(dirpath,filename)
     409             archivename = os.path.normpath(templatefile)
     410             archivename = '/'.join(archivename.split(os.sep))
     411             if archivename in files_to_skip:
     412                 continue
     413             #print 'Saving: '+archivename          
     414             docxfile.write(templatefile, archivename)
     415 
     416     os.chdir(prev_dir) # restore previous working dir
     417     return docxfile






How is styleDocx used ?
--------------------------

Its central to operation.


Every Line of RST source is becoming a paragraph
---------------------------------------------------

* rst blank lines not being honoured as paragraph separators ? Nope
  seems every line of RST source is becoming a paragraph despite 
  add_text being called for each intended para so intended splitting info 
  is being lost subsequently

  * essentially RST newlines are incorrectly causing new paragraphs

::

    2015-04-24 15:28:10,348 docxbuilder.writer WARNING  add_text [Incoming neutrinos interact weakly with materials by the inverse beta decay process
    where an electron anti-neutrino is captured on a proton of the target resulting
    in the production of a positron and a neutron.
    The positron yields a prompt energy deposit followed by a delayed
    energy deposit from the neutron.
    Detector materials and geometry are chosen to enable detection of these energy deposits.]
    2015-04-24 15:28:10,349 docxbuilder.writer WARNING  add_text [The majority of neutrino detectors operate by the detection of visible light produced
    in transparent target materials such as water, ice or liquid scintillator.
    The weak nature of the neutrino interactions typically necessitate large volumes
    of material and large numbers of Photo Multiplier Tubes (PMT) to detect the light.]


kludge it by replacing newlines with spaces within docx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    0999   def make_run(self, txt, style='Normal', rawXml=None):
    1000     '''
    1001       Make a new styled run from text.
    1002     '''
    1003     log.fatal("make_run ..%s.. " % txt )
    1004     run_tree = [['w:r']]
    1005     if txt == ":br" :
    1006       run_tree.append([['w:br']])
    1007     else:
    1008       attr ={}
    1009       if txt.find(' ') != -1 :
    1010         attr ={'xml:space':'preserve'}
    1011 
    1012       if style != 'Normal' :
    1013         if style not in self.stylenames :
    1014           self.new_character_style(style)
    1015 
    1016     run_tree.append([['w:rPr'], [['w:rStyle',{'w:val':style}], [['w:t', txt, attr]] ]])
    1017       else:
    1018         ttxt = txt.replace("\n", " ") # SCB
    1019         run_tree.append([['w:t', ttxt, attr]])
    1020 


Style names
-------------

* https://msdn.microsoft.com/en-us/library/documentformat.openxml.wordprocessing.stylename(v=office.14).aspx

Messy near equivalents::

    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL insert_paragraph_property new_paragraph_style Heading2 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k         Default Paragraph Font v         Default Paragraph Font 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k                 DefinitionItem v                 DefinitionItem 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k                Header & Footer v                Header & Footer 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k                        Heading v                        Heading 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k                       Heading1 v                       Heading1 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k                       Heading4 v                       Heading4 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k                       Heading5 v                       Heading5 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k                      Hyperlink v                      Hyperlink 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k                   ImageCaption v                   ImageCaption 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k               Imported Style 1 v               Imported Style 1 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k               Imported Style 2 v               Imported Style 2 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k                         List 0 v                         List 0 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k                 List Paragraph v                 List Paragraph 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k                     ListBullet v                     ListBullet 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k                     ListNumber v                     ListNumber 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k                   LiteralBlock v                   LiteralBlock 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k                        No List v                        No List 
    2015-04-24 17:04:41,302 docxbuilder.docx.docx CRITICAL  k                         Normal v                         Normal 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                     Subtitle A v                     Subtitle A 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                          TOC 2 v                          TOC 2 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                          TOC 3 v                          TOC 3 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                          TOC 4 v                          TOC 4 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                          TOC 5 v                          TOC 5 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                          TOC 6 v                          TOC 6 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                          TOC 7 v                          TOC 7 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                          TOC 8 v                          TOC 8 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                          TOC 9 v                          TOC 9 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                   TOC_Contents v                   TOC_Contents 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                   Table Normal v                   Table Normal 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                   TableHeading v                   TableHeading 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                        Title A v                        Title A 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                      heading 1 v                      heading 1 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                      heading 2 v                      Heading 2 
    2015-04-24 17:04:41,303 docxbuilder.docx.docx CRITICAL  k                      heading 3 v                      Heading 3 

Running with original style.docx suggests braindead Pages 5.2 is responsible for the messiness.


Unbelievable microsoft s**t exploding zip::

    delta:docx blyth$ cp style.docx style_docx.zip
    delta:docx blyth$ unzip style_docx.zip -d style_docx
    Archive:  style_docx.zip
      inflating: style_docx/[Content_Types].xml  
      inflating: style_docx/_rels/.rels  
      inflating: style_docx/word/_rels/document.xml.rels  
      inflating: style_docx/word/document.xml  
      inflating: style_docx/word/footnotes.xml  
      inflating: style_docx/word/endnotes.xml  
      inflating: style_docx/word/theme/theme1.xml  
      inflating: style_docx/word/settings.xml  
      inflating: style_docx/word/glossary/_rels/document.xml.rels  
      inflating: style_docx/word/glossary/settings.xml  
      inflating: style_docx/word/glossary/document.xml  
      inflating: style_docx/word/glossary/styles.xml  
      inflating: style_docx/docProps/app.xml  
      inflating: style_docx/word/styles.xml  
      inflating: style_docx/word/webSettings.xml  
      inflating: style_docx/word/stylesWithEffects.xml  
      inflating: style_docx/word/fontTable.xml  
      inflating: style_docx/word/glossary/fontTable.xml  
      inflating: style_docx/word/glossary/webSettings.xml  
      inflating: style_docx/word/numbering.xml  
      inflating: style_docx/word/glossary/stylesWithEffects.xml  
      inflating: style_docx/docProps/core.xml  


::

    delta:word blyth$ xmllint --pretty 1  styles.xml | grep \<w:style
    <w:styles xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" mc:Ignorable="w14">
      <w:style w:type="paragraph" w:default="1" w:styleId="a">
      <w:style w:type="paragraph" w:styleId="1">
      <w:style w:type="paragraph" w:styleId="2">
      <w:style w:type="paragraph" w:styleId="3">
      <w:style w:type="paragraph" w:styleId="4">
      <w:style w:type="paragraph" w:styleId="5">
      <w:style w:type="character" w:default="1" w:styleId="a0">
      <w:style w:type="table" w:default="1" w:styleId="a1">
      <w:style w:type="numbering" w:default="1" w:styleId="a2">
      <w:style w:type="character" w:customStyle="1" w:styleId="10">
      <w:style w:type="paragraph" w:styleId="a3">
      <w:style w:type="character" w:customStyle="1" w:styleId="a4">
      <w:style w:type="paragraph" w:styleId="a5">
      <w:style w:type="character" w:customStyle="1" w:styleId="a6">
      <w:style w:type="character" w:customStyle="1" w:styleId="20">
      <w:style w:type="paragraph" w:customStyle="1" w:styleId="Heading1">
      <w:style w:type="paragraph" w:customStyle="1" w:styleId="Heading2">
      <w:style w:type="paragraph" w:customStyle="1" w:styleId="Heading3">
      <w:style w:type="paragraph" w:customStyle="1" w:styleId="Heading4">



Strategy 
----------

* docxbuilder incorporates an old docx, 
* docxbuilder aint that complicated as it has a nice regular doctree to work from. 

Potentially would be easier to update to latest docx ? Before getting too stuck into 
debugging the old docx within docxbuilder.



Usage
-------

#. add module to python with, docxbuilder-ln
#. add extension to sphinx by adding to eg workflow/admin/reps/conf.py 

     33 # Add any Sphinx extension module names here, as strings. They can be extensions
     34 # coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
     35 extensions = ['sphinx.ext.todo', 'sphinx.ext.ifconfig', 'sphinx.ext.extlinks', 'docxbuilder' ]

#. add docx phony target to Makefile

#. try *make docx*

   



EOU
}
docxbuilder-dir(){ echo $(local-base)/env/tools/docxbuilder ; }
docxbuilder-cd(){  cd $(docxbuilder-dir); }
docxbuilder-mate(){ mate $(docxbuilder-dir) ; }
docxbuilder-get(){
   local dir=$(dirname $(docxbuilder-dir)) &&  mkdir -p $dir && cd $dir

   hg clone https://bitbucket.org/haraisao/sphinx-docxbuilder docxbuilder
}

docxbuilder-ln(){
   python-
   python-ln $(docxbuilder-dir)
}


