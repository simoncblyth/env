/*

* http://stackoverflow.com/questions/4131225/invoke-a-unix-shell-from-java-programread-and-write-a-steady-stream-of-data-to
* :google:`Runtime.getRuntime()`
* http://docs.oracle.com/javase/6/docs/api/java/lang/Process.html
* http://alvinalexander.com/java/edu/pj/pj010016

* http://commons.apache.org/proper/commons-exec/

Usage::

    javac RuntimeExec.java
    java RuntimeExec      # succeeds to cat ~/.bash_profile

*/

import java.io.*;

class RuntimeExec {
    public static void main(String[] args) {
        String s = null;
        try {

            String[] cmd = {"/bin/sh", "-c", "/bin/cat /Users/heprez/.bash_profile "};
            Process p = Runtime.getRuntime().exec(cmd);
            OutputStream out = p.getOutputStream();  // to process stdin

            BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));  // process stdout
            BufferedReader stdError = new BufferedReader(new InputStreamReader(p.getErrorStream()));  // process stderr

            while ((s = stdInput.readLine()) != null) {
                    System.out.println(s);
                }

            while ((s = stdError.readLine()) != null) {
                    System.out.println(s);
                }

            System.exit(0);
        }
        catch (IOException e) {
            System.out.println("exception happened - here's what I know: ");
            e.printStackTrace();
            System.exit(-1);
        }

    }
}


