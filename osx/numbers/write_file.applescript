on write_file(the_path, the_text)

    set the_lf to ASCII character 10
    set the_file to open for access POSIX file the_path with write permission 
    write (the_text & the_lf) to the_file 
    close access the_file
    
end write_file


on run argv

   set the_path to (item 1 of argv)
   write_file( the_path, "hello")

end run 




