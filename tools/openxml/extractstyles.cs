// Extract the styles or stylesWithEffects part from a 
// word processing document as an XDocument instance.


using System;
using System.IO;
using System.Xml;
using System.Xml.Linq;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;




class Program
{

   public static XDocument ExtractStylesPart( string fileName, bool getStylesWithEffectsPart = true)
   {
       XDocument styles = null;

       //var isEditable = false ;  //  gives: Operation not valid when package is read-only
       var isEditable = true ; 

       using (var document = WordprocessingDocument.Open(fileName, isEditable))
       {
            var docPart = document.MainDocumentPart;

            StylesPart stylesPart = null;
            if (getStylesWithEffectsPart)
                stylesPart = docPart.StylesWithEffectsPart;
            else
                stylesPart = docPart.StyleDefinitionsPart;

            if (stylesPart != null)
            {
                using (var reader = XmlNodeReader.Create( stylesPart.GetStream(FileMode.Open, FileAccess.Read)))
                {
                    styles = XDocument.Load(reader);
                }
            }
       }
       return styles;
    }


    static void Main(string[] args)
    {

        if (args.Length > 0)
        {
                Console.WriteLine(args[0]);

                //var withEffects = true ; 
                var withEffects = false ; 
                var styles = ExtractStylesPart(args[0], withEffects );
                if (styles != null)
                    Console.WriteLine(styles.ToString());
        }
    }
}


