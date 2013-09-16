/*

Usage::


*/

import java.io.*;
import org.apache.commons.cli.*;

class SceneEdit {

    float[] diffuseColor = null ; 
    float[] emissiveColor = null ; 
    float[] specularColor = null ; 
    float[] shininess = null ; 
    float[] transparency = null ; 
    float[] ambientIntensity = null ; 
    String[] nodenames = null ; 

    public SceneEdit()
    {
    } 

    public static void main(String[] args) throws Exception 
    {
        SceneEdit edit = null ;
		vrml.eai.Browser browser = null;
	    java.net.InetAddress address = java.net.InetAddress.getByName("localhost");

        try {
			browser = vrml.eai.BrowserFactory.getBrowser(address, 4848);
            dumpBrowser( browser );
            edit = constructSceneEdit( args );
            applyEdit( browser, edit );
		}
		catch (Throwable all)
		{
			all.printStackTrace();
		}
		finally
		{
			if (browser != null) browser.dispose();
		}
	}


    public static void dumpBrowser( vrml.eai.Browser browser )
    {
		System.out.println("Browser.Name = \"" + browser.getName() + '"');
		System.out.println("Browser.Version = \"" + browser.getVersion() + '"');
		System.out.println("Browser.CurrentSpeed = " + browser.getCurrentSpeed());
		System.out.println("Browser.CurrentFrameRate = " + browser.getCurrentFrameRate());
		System.out.println("Browser.WorldURL = \"" + browser.getWorldURL() + '"');
    } 

    public static SceneEdit constructSceneEdit(String[] args)
    {
        SceneEdit se = null ;
        Options options = constructOptions();
        CommandLineParser parser = new PosixParser();
        try {
            CommandLine cli = parser.parse( options, args );
            se = SceneEdit.create( cli );
        }
        catch( ParseException exp ) {
            System.err.println( "Parsing failed.  Reason: " + exp.getMessage() );
        }
        return se ;
    }

	public static void applyEdit(vrml.eai.Browser browser, SceneEdit edit)
	{
         browser.beginUpdate();
         for(int i = 0 ; i < edit.nodenames.length ; i++ )
         {
             String name = edit.nodenames[i];
             vrml.eai.Node node = browser.getNode(name);
             applyEdit( node, edit, name ); 
         }
         browser.endUpdate();
    }


    public static void changeColor( vrml.eai.Node material, String attname, float[] color )
    {
         assert(color.length == 3 ); 
         vrml.eai.field.EventOutSFColor colorOut = (vrml.eai.field.EventOutSFColor)material.getEventOut(attname + "_changed");
         vrml.eai.field.EventInSFColor colorIn = (vrml.eai.field.EventInSFColor)material.getEventIn("set_" + attname);
         float[] priorColor = colorOut.getValue();
         colorIn.setValue(color);
    } 

    public static void changeFloat( vrml.eai.Node material, String attname, float[] attv )
    {
         assert(attv.length == 1 ); 
         vrml.eai.field.EventOutSFFloat attOut = (vrml.eai.field.EventOutSFFloat)material.getEventOut(attname + "_changed");
         vrml.eai.field.EventInSFFloat attIn = (vrml.eai.field.EventInSFFloat)material.getEventIn("set_" + attname);
         float priorAtt = attOut.getValue();
         System.out.println("changeFloat " + attname + " from " + priorAtt + " to " + attv[0] );
         attIn.setValue(attv[0]);
    } 

    public static void applyEdit( vrml.eai.Node node, SceneEdit edit , String name)
    {
         String type = node.getType();
         System.out.println( "applyEdit to node : " + name + " type : [" + type + "] : " + node );

         if(!type.equals("Material")){
             System.out.println( "applyEdit SKIP NON-Material node : " + name + " type : " + type + " : " + node );
             return ;
         } 

         if( edit.diffuseColor != null )  changeColor(node, "diffuseColor", edit.diffuseColor);
         if( edit.emissiveColor != null ) changeColor(node, "emissiveColor", edit.emissiveColor);
         if( edit.specularColor != null ) changeColor(node, "specularColor", edit.specularColor);
         if( edit.transparency != null ) changeFloat(node, "transparency", edit.transparency );
         if( edit.shininess != null ) changeFloat(node, "shininess", edit.shininess );
         if( edit.ambientIntensity != null ) changeFloat(node, "ambientIntensity", edit.ambientIntensity );
    } 


    public static SceneEdit create( CommandLine cli)
    {
        SceneEdit se = new SceneEdit() ;

        se.diffuseColor = interpretOption(cli,"diffuseColor");
        se.emissiveColor = interpretOption(cli,"emissiveColor");
        se.specularColor = interpretOption(cli,"specularColor");
        se.transparency = interpretOption(cli,"transparency");
        se.shininess = interpretOption(cli,"shininess");
        se.ambientIntensity= interpretOption(cli,"ambientIntensity");
        se.nodenames = cli.getArgs(); 
        return se ;
    } 

    public void dump()
    {
         dumpFloats( this.diffuseColor , "diffuseColor" );
         dumpFloats( this.emissiveColor , "emissiveColor" );
         dumpFloats( this.specularColor , "specularColor" );
         dumpFloats( this.transparency, "transparency" );
         dumpFloats( this.shininess , "shininess" );
         dumpFloats( this.ambientIntensity, "ambientIntensity" );
         dumpStrings( this.nodenames, "nodenames" );
    }


    public static Options constructOptions()  
    {  
         final Options options = new Options();  
         options.addOption("address", true, "inet address of reality player ");  
         options.addOption("port", true, "inet port");  

         options.addOption("diffuseColor", true, "color specifier eg \"1,0,0\" or \"0.5,0.5,0\" ");  
         options.addOption("emissiveColor", true, "color specifier eg \"1,0,0\" or \"0.5,0.5,0\" ");  
         options.addOption("specularColor", true, "color specifier eg \"1,0,0\" or \"0.5,0.5,0\" ");  
         options.addOption("transparency", true, "float specified eg \"0.7\" ");
         options.addOption("shininess", true, "float specified eg \"0.7\" ");
         options.addOption("ambientIntensity", true, "float specified eg \"0.7\" ");
         return options;  
    }  

    public static float[] interpretOption(CommandLine cli, String tag)
    {
        String opt = cli.getOptionValue(tag);
        return parseFloats(opt);
    }

    public static float[] parseFloats(String sf)
    {
         float[] fa = null ; 
         if( sf == null) return fa ; 
         String[] sfe = sf.split(",");
         fa = new float[sfe.length];
         for (int i = 0 ; i < sfe.length; i++) fa[i] = Float.parseFloat(sfe[i]);
         return fa ;
    } 

    public static void dumpStrings( String[] fa, String name)
    {
        if ( fa == null ){
             System.out.println(name + " [null] "  );
        } else if( fa.length == 0 ){
             System.out.println(name + " []" );
        } else if( fa.length > 0 ){
             System.out.println(name + " [" + fa.length + "]" );
             for(int i = 0 ; i < fa.length ; i++ ) System.out.println( i + " " + fa[i] );
        }
    }

    public static void dumpFloats( float[] fa, String name)
    {
        if ( fa == null ){
             System.out.println(name + " [null] "  );
        } else if( fa.length == 0 ){
             System.out.println(name + " []" );
        } else if( fa.length == 1 ){
             System.out.println(name + " [" + fa[0] + "]" );
        } else if( fa.length == 3 ){
             System.out.println(name + " [" + fa[0] + "," + fa[1] + "," + fa[2] + "]" );
        } else {
             System.out.println(name + " UNEXPECTED LENGTH : " + fa.length  );
        }
    }

}






