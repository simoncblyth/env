/*


* http://commons.apache.org/proper/commons-exec/
* http://stackoverflow.com/questions/7340452/process-output-from-apache-commons-exec

Usage::

    javac -cp $(jexec-jar) CommonsExec.java && java -cp .:$(jexec-jar) CommonsExec

*/

import java.io.*;
import org.apache.commons.exec.*;

class CommonsExec {
    public static void main(String[] args) throws Exception 
    {
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        ByteArrayOutputStream stderr = new ByteArrayOutputStream();
        //ByteArrayOutputStream stdout = new ByteArrayOutputStream();

        //PipedOutputStream stdout = new PipedOutputStream();
        //PipedOutputStream stderr = new PipedOutputStream();
        //PipedInputStream stdin = new PipedInputStream();

        PumpStreamHandler psh = new PumpStreamHandler(stdout, stderr);

        String cmd = null ; 

        cmd = "perl -V" ; 
        //cmd = "env" ;  
        //cmd = "perl -MStorable -MData::Dumper -e 'print Dumper(retrieve $ARGV[0]);' /Users/heprez/data/data/images/pdgparse/pdg.store " ;  
        //cmd = "perl -MStorable -MData::Dumper -e 'print \"hello\";' " ;  

        CommandLine cl = CommandLine.parse(cmd);
        DefaultExecutor exec = new DefaultExecutor();
        exec.setStreamHandler(psh);
        exec.execute(cl);

        System.out.println(stdout.toString());
        System.out.println(stderr.toString());
   }
}


