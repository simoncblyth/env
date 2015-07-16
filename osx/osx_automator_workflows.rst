OSX Automator Workflows
========================

.. contents:: :local:


Automator Service to Make PDF from Multiple PNGs
-------------------------------------------------

Refs 
~~~~~~

* http://miriamposner.com/blog/use-automator-to-combine-your-research-photos-into-one-pdf/



Using *Make PDF from PNGs* Automator Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#. Open folder containing PNGs to combine in Finder.app
#. Arrange the order and then select PNGs
#. `ctrl-click` the selection to get contextual menu, choose `Make PDF from PNGs`
#. After a second or so (depending on number of PNGs) a dialog will appear to enter the 
   basename (without .pdf) of the output PDF. 
#. After entering the name the new PDF will appear on the Desktop


Creating *Make PDF from PNGs* Automator Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#. Open Automator, choose *Service* 

   * Select `Service receives: image files` in `any application`
   * Drag `Library > PDFs > New PDF from Images` action from 2nd column into empty 3rd column
   * Drag `Library > Files & Folders > Rename Finder Items` 

     * click through the warning as this is a new file 
     * select `Name Single Item` from the list and in `Options` tab select `Show this action when the workflow runs`

   * Save workflow as `Make PDF from PNGs` and exit Automator.app


Modifying *Make PDF from PNGs* Automator Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Find it::

    delta:~ blyth$ mdfind "Make PDF from PNGs.workflow"
    /Users/blyth/Library/Services/Make PDF from PNGs.workflow
    delta:~ blyth$ 


Page Sizing
~~~~~~~~~~~~~

Existing PDF from PNGs is making all pages 1024x768





Automator Service to Combine Multiple PDFs 
--------------------------------------------

Using *Combine PDFs* Automator Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#. Open a folder containing PDFs to combine in Finder.app.
#. Adjust the Finder order as desired
#. select the PDFs to be combined, and `ctrl-click` on them, 
   the `Combine PDFs` service should appear beneath Tags.

#. a `Rename Finder Items` dialog will appear, select a name (basename only, no .pdf extension)
   and hit **Continue** and the combined PDF should appear on Desktop


Creating *Combine PDFs* Automator Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow along:

* http://www.documentsnap.com/how-to-combine-pdf-files-in-mac-osx-using-automator-to-make-a-service/


#. Open Automator, choose *Service*, 

   * Select `Service receives: PDF files`  in `any application`
   * Drag `Library > PDFs > Combine PDF Pages` action from 2nd column into the empty 3rd column
   * Drag `Library > Files & Folders > Rename Finder Items`

     * a warning comes up, about using a copy instead of a rename. Its OK however 
       as we are dealing with a newly created PDF, so agree to the rename

     * select `Name Single Item` from the list and in `Options` tab select `Show this action when the workflow runs`

   * NB to give more space in 3rd column, close the disclosure triangles on the actions 
    
   * Drag `Library > Files & Folders > Move Finder Items` into the 3rd column
 
     * leave destination at default location of `Desktop`

#. Now `File > Save..` and enter name for the Service: `Combine PDFs` and exit from Automator.app

   * NB exiting is necessary, as without it the Service does not appear in contextual menu
     after having selected some PDFs












