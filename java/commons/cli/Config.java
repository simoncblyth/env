/*

Usage::

    simon:cli blyth$ jcli-demo -diffuseColor 0.5,0.5,0.1 -emissiveColor 1,1,1 -specularColor 1,0,0 -transparency 0.6 -shininess 0.5                      
    diffuseColor [0.5,0.5,0.1]
    emissiveColor [1.0,1.0,1.0]
    specularColor [1.0,0.0,0.0]
    transparency [0.6]
    shininess [0.5]
    ambientIntensity [null] 


*/

import java.io.*;
import org.apache.commons.cli.*;

class Config {

    float[] diffuseColor = null ; 
    float[] emissiveColor = null ; 
    float[] specularColor = null ; 
    float[] shininess = null ; 
    float[] transparency = null ; 
    float[] ambientIntensity = null ; 

    public Config()
    {
    } 

    public static Config createConfig( CommandLine cli)
    {
        Config cfg = new Config() ;
        cfg.diffuseColor = interpretOption(cli,"diffuseColor");
        cfg.emissiveColor = interpretOption(cli,"emissiveColor");
        cfg.specularColor = interpretOption(cli,"specularColor");
        cfg.transparency = interpretOption(cli,"transparency");
        cfg.shininess = interpretOption(cli,"shininess");
        cfg.ambientIntensity= interpretOption(cli,"ambientIntensity");
        return cfg ;
    } 

    public static Options constructOptions()  
    {  
         final Options options = new Options();  
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

    public static void main(String[] args) throws Exception 
    {
        Options options = constructOptions();
        CommandLineParser parser = new PosixParser();
        try {
            CommandLine cli = parser.parse( options, args );
            Config cfg = Config.createConfig( cli );
            cfg.dump();
        }
        catch( ParseException exp ) {
            System.err.println( "Parsing failed.  Reason: " + exp.getMessage() );
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

    public void dump()
    {
         dumpFloats( this.diffuseColor , "diffuseColor" );
         dumpFloats( this.emissiveColor , "emissiveColor" );
         dumpFloats( this.specularColor , "specularColor" );
         dumpFloats( this.transparency, "transparency" );
         dumpFloats( this.shininess , "shininess" );
         dumpFloats( this.ambientIntensity, "ambientIntensity" );
    }
}


