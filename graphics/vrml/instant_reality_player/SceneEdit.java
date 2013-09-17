/*

Usage::


*/

import java.io.*;
import org.apache.commons.cli.*;

class SceneEdit {

    String[] address = null ;
    String[] port = null ;

    // parameters apply to the named viewpoint nodes
    String[] viewpoints = null ;
    float[] position = null ;
    float[] upVector = null ;
    float[] centerOfRotation = null ;
    float[] orientation = null ;

    // properties are applied to all the named material nodes 
    String[] nodenames = null ; 
    float[] diffuseColor = null ; 
    float[] emissiveColor = null ; 
    float[] specularColor = null ; 
    float[] shininess = null ; 
    float[] transparency = null ; 
    float[] ambientIntensity = null ; 

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

         if(edit.viewpoints != null )
         {
             for(int i = 0 ; i < edit.viewpoints.length ; i++ )
             {
                 String v = "V" + edit.viewpoints[i];
                 vrml.eai.Node vnode = browser.getNode(v);
                 applyViewpointEdit( vnode, edit, v );
             }
         }
         
         if(edit.nodenames != null ) 
         {
             for(int i = 0 ; i < edit.nodenames.length ; i++ )
             {
                 String m = "M" + edit.nodenames[i];
                 vrml.eai.Node mnode = browser.getNode(m);
                 applyMaterialEdit( mnode, edit, m ); 
             }
         } 

         browser.endUpdate();
    }

    public static void checkSFVec3f( vrml.eai.Node node, String attname)
    {
        vrml.eai.field.EventOutSFVec3f vecOut = (vrml.eai.field.EventOutSFVec3f)node.getEventOut(attname);
        float[] val = vecOut.getValue();
        dumpFloats( val, "checkSFVec3f " + attname );
    }

    public static void checkSFRotation( vrml.eai.Node node, String attname)
    {
        vrml.eai.field.EventOutSFRotation vecOut = (vrml.eai.field.EventOutSFRotation)node.getEventOut(attname);
        float[] val = vecOut.getValue();
        dumpFloats( val, "checkSFRotation " + attname );
    }

    public static void changeSFRotation( vrml.eai.Node node, String attname, float[] vec )
    {
        vrml.eai.field.EventInSFRotation vecIn = (vrml.eai.field.EventInSFRotation)node.getEventIn("set_" + attname );
        dumpFloats(vec, "changeSFRotation " + attname );
        vecIn.setValue(vec);
    }

    public static void changeSFVec3f( vrml.eai.Node node, String attname, float[] vec )
    {
        vrml.eai.field.EventInSFVec3f vecIn = (vrml.eai.field.EventInSFVec3f)node.getEventIn("set_" + attname );
        dumpFloats(vec, "changeSFVec3f " + attname );
        vecIn.setValue(vec);
    }

    public static void changeSFColor( vrml.eai.Node material, String attname, float[] color )
    {
         assert(color.length == 3 ); 
         vrml.eai.field.EventOutSFColor colorOut = (vrml.eai.field.EventOutSFColor)material.getEventOut(attname + "_changed");
         vrml.eai.field.EventInSFColor colorIn = (vrml.eai.field.EventInSFColor)material.getEventIn("set_" + attname);
         float[] priorColor = colorOut.getValue();
         colorIn.setValue(color);
    } 

    public static void changeSFFloat( vrml.eai.Node material, String attname, float[] attv )
    {
         assert(attv.length == 1 ); 
         vrml.eai.field.EventOutSFFloat attOut = (vrml.eai.field.EventOutSFFloat)material.getEventOut(attname + "_changed");
         vrml.eai.field.EventInSFFloat attIn = (vrml.eai.field.EventInSFFloat)material.getEventIn("set_" + attname);
         float priorAtt = attOut.getValue();
         System.out.println("changeFloat " + attname + " from " + priorAtt + " to " + attv[0] );
         attIn.setValue(attv[0]);
    } 

    public static void applyViewpointEdit( vrml.eai.Node node, SceneEdit edit , String name)
    {

         /*
             http://doc.instantreality.org/apidocs/scripting/javascript/classSFMatrix4f.html
         */
         String type = node.getType();
         if(!type.equals("Viewpoint")){
             System.out.println( "applyViewpointEdit SKIP NON-Viewpoint node : " + name + " type : " + type + " : " + node );
             return ;
         } 

         checkSFVec3f(node, "position");
         checkSFVec3f(node, "upVector");
         checkSFVec3f(node, "centerOfRotation");
         checkSFRotation(node, "orientation");

         if( edit.position != null )  changeSFVec3f(node, "position", edit.position );
         if( edit.upVector != null )  changeSFVec3f(node, "upVector", edit.upVector );
         if( edit.centerOfRotation != null )  changeSFVec3f(node, "centerOfRotation", edit.centerOfRotation );
         if( edit.orientation != null )  changeSFRotation(node, "orientation", edit.orientation );

    }

    public static void applyMaterialEdit( vrml.eai.Node node, SceneEdit edit , String name)
    {
         String type = node.getType();
         System.out.println( "applyEdit to node : " + name + " type : [" + type + "] : " + node );
         if(!type.equals("Material")){
             System.out.println( "applyEdit SKIP NON-Material node : " + name + " type : " + type + " : " + node );
             return ;
         } 

         if( edit.diffuseColor != null )  changeSFColor(node, "diffuseColor", edit.diffuseColor);
         if( edit.emissiveColor != null ) changeSFColor(node, "emissiveColor", edit.emissiveColor);
         if( edit.specularColor != null ) changeSFColor(node, "specularColor", edit.specularColor);
         if( edit.transparency != null ) changeSFFloat(node, "transparency", edit.transparency );
         if( edit.shininess != null ) changeSFFloat(node, "shininess", edit.shininess );
         if( edit.ambientIntensity != null ) changeSFFloat(node, "ambientIntensity", edit.ambientIntensity );
    } 


    public static SceneEdit create( CommandLine cli)
    {
        SceneEdit se = new SceneEdit() ;

        se.viewpoints = interpretStringOption(cli, "viewpoints");
        se.nodenames = cli.getArgs(); 

        se.position = interpretOption(cli, "position" );
        se.upVector = interpretOption(cli, "upVector" );
        se.centerOfRotation = interpretOption(cli, "centerOfRotation");
        se.orientation = interpretOption(cli, "orientation");

        se.diffuseColor = interpretOption(cli,"diffuseColor");
        se.emissiveColor = interpretOption(cli,"emissiveColor");
        se.specularColor = interpretOption(cli,"specularColor");
        se.transparency = interpretOption(cli,"transparency");
        se.shininess = interpretOption(cli,"shininess");
        se.ambientIntensity= interpretOption(cli,"ambientIntensity");

        return se ;
    } 

    public void dumpViewpointEdits()
    {
         dumpFloats( this.orientation , "orientation" );
         dumpFloats( this.position , "position" );
         dumpFloats( this.upVector , "upVector" );
         dumpFloats( this.centerOfRotation , "centerOfRotation" );
         dumpStrings( this.viewpoints, "viewpoints" );
    }

    public void dumpMaterialEdits()
    {
         dumpFloats( this.diffuseColor , "diffuseColor" );
         dumpFloats( this.emissiveColor , "emissiveColor" );
         dumpFloats( this.specularColor , "specularColor" );
         dumpFloats( this.transparency, "transparency" );
         dumpFloats( this.shininess , "shininess" );
         dumpFloats( this.ambientIntensity, "ambientIntensity" );
         dumpStrings( this.nodenames, "nodenames" );
    }

    public void dump()
    {
        dumpViewpointEdits();
        dumpMaterialEdits();
    }

    public static Options constructOptions()  
    {  
         final Options options = new Options();  

         options.addOption("address", true, "inet address of reality player ");  
         options.addOption("port", true, "inet port");  
         options.addOption("viewpoints", true, "Comma delimited viewpoint node name eg \"0,1\" corresponding to nodes \"V0\" and \"V1\"  ");  

         options.addOption("position", true, "3f Viewpoint position specifier eg \"0,0,111242\"  ");  
         options.addOption("upVector", true, "3f Viewpoint position specifier eg \"0,0,111242\"  ");  
         options.addOption("centerOfRotation", true, "3f Viewpoint position specifier eg \"0,0,111242\"  ");  
         options.addOption("orientation", true, "4f Viewpoint position specifier eg \"0,0,111242\"  ");  

         options.addOption("diffuseColor", true, "color specifier eg \"1,0,0\" or \"0.5,0.5,0\" ");  
         options.addOption("emissiveColor", true, "color specifier eg \"1,0,0\" or \"0.5,0.5,0\" ");  
         options.addOption("specularColor", true, "color specifier eg \"1,0,0\" or \"0.5,0.5,0\" ");  
         options.addOption("transparency", true, "float specified eg \"0.7\" ");
         options.addOption("shininess", true, "float specified eg \"0.7\" ");
         options.addOption("ambientIntensity", true, "float specified eg \"0.7\" ");
         return options;  
    }  

    public static String[] interpretStringOption(CommandLine cli, String tag)
    {
        String opt = cli.getOptionValue(tag);
        if( opt == null ) return null ;
        return opt.split(",");
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
        } else if( fa.length == 4 ){
             System.out.println(name + " [" + fa[0] + "," + fa[1] + "," + fa[2] + "," + fa[3] + "]" );
        } else {
             System.out.println(name + " UNEXPECTED LENGTH : " + fa.length  );
        }
    }

}






