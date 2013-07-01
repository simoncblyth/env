/*

* http://commons.apache.org/proper/commons-exec/
* http://stackoverflow.com/questions/7340452/process-output-from-apache-commons-exec

Usage::

    javac -cp $(jexec-jar) CommonsExec.java && java -cp .:$(jexec-jar)  -Dscript.path=/Users/heprez/e/java/commons/exec/demo.pl -Dscript.args="red green blue yellow purple" CommonsExec


Note 

#. property settings have to come before the classname on commandline

*/

import java.io.*;
import org.apache.commons.exec.*;

class CommonsExec {
    public static void main(String[] args) throws Exception 
    {
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        ByteArrayOutputStream stderr = new ByteArrayOutputStream();
        PumpStreamHandler psh = new PumpStreamHandler(stdout, stderr);

        String script_path = System.getProperty("script.path", "" );
        String script_args = System.getProperty("script.args", "" );
        String cmd = script_path + " " + script_args ; 

        System.out.println("CommonsExec : " + cmd );

        CommandLine cl = CommandLine.parse(cmd);
        DefaultExecutor exec = new DefaultExecutor();
        exec.setStreamHandler(psh);
        exec.execute(cl);

        System.out.println(stdout.toString());
        System.out.println(stderr.toString());
   }
}


