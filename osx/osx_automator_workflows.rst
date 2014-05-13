OSX Automator Workflows
========================

.. contents:: :local:

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


#. Open Automator, Choose *Service*, 

   * Select `Service received: PDF files`  in `any application`
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












