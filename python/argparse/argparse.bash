# === func-gen- : python/argparse/argparse fgp python/argparse/argparse.bash fgn argparse fgh python/argparse
argparse-src(){      echo python/argparse/argparse.bash ; }
argparse-source(){   echo ${BASH_SOURCE:-$(env-home)/$(argparse-src)} ; }
argparse-vi(){       vi $(argparse-source) ; }
argparse-env(){      elocal- ; }
argparse-usage(){ cat << EOU

argparse
==========

Bizarre argparse non-empty whitespace help bug
-------------------------------------------------

OK::

   parser.add_argument("-i","--ipython", action="store_true", help=""  )  

Triggers the bug:: 

   parser.add_argument("-i","--ipython", action="store_true", help=" "  )
   parser.add_argument("-i","--ipython", action="store_true", help="  "  )
   parser.add_argument("-i","--ipython", action="store_true", help="   "  )  


Looking like::

    delta:pubs blyth$ pubs.py --help
    Traceback (most recent call last):
      File "/Users/blyth/workflow/bin/pubs.py", line 4, in <module>
        main()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/workflow/admin/pubs/pubs.py", line 193, in main
        args = parse(__doc__) 
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/workflow/admin/pubs/pubs.py", line 98, in parse
        args = parser.parse_args()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/argparse.py", line 1701, in parse_args
        args, argv = self.parse_known_args(args, namespace)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/argparse.py", line 1733, in parse_known_args
        namespace, args = self._parse_known_args(args, namespace)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/argparse.py", line 1939, in _parse_known_args
        start_index = consume_optional(start_index)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/argparse.py", line 1879, in consume_optional
        take_action(action, args, option_string)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/argparse.py", line 1807, in take_action
        action(self, namespace, argument_values, option_string)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/argparse.py", line 996, in __call__
        parser.print_help()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/argparse.py", line 2340, in print_help
        self._print_message(self.format_help(), file)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/argparse.py", line 2314, in format_help
        return formatter.format_help()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/argparse.py", line 281, in format_help
        help = self._root_section.format_help()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/argparse.py", line 211, in format_help
        func(*args)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/argparse.py", line 211, in format_help
        func(*args)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/argparse.py", line 519, in _format_action
        parts.append('%*s%s\n' % (indent_first, '', help_lines[0]))
    IndexError: list index out of range




EOU
}
argparse-dir(){ echo $(local-base)/env/python/argparse/python/argparse-argparse ; }
argparse-cd(){  cd $(argparse-dir); }
argparse-mate(){ mate $(argparse-dir) ; }
argparse-get(){
   local dir=$(dirname $(argparse-dir)) &&  mkdir -p $dir && cd $dir

}
