(*
Usage::

    osascript numbers_export_csv.applescript $PWD/export.csv "Journal paper" 3
 
Writes to a csv file specified by an absolute unix path 
the values extracted from the named worksheet of the frontmost 
spreadsheet opened in OSX Numbers.app starting with 
the row numbered in the third argument (1-based).

*)


on write_file(the_path, the_text)
    
    set the_lf to ASCII character 10
    set the_file to open for access POSIX file the_path with write permission
    write (the_text & the_lf) to the_file
    close access the_file
    
end write_file


on join_items(the_items, the_delimiter)

    set {tids, text item delimiters} to {text item delimiters, the_delimiter}
    set the_string to the_items as string
    set text item delimiters to tids
    return the_string

end join_items


on export_csv( the_csvpath, the_sheetname, the_toprow, the_delimiter )

    set the_lines to {}

    tell application "Numbers" to tell document 1 to tell sheet the_sheetname to tell table 1
        repeat with the_row_index from the_toprow to (row count)
            tell row the_row_index
                set the_values to (value of cells)
                set the_line to my join_items(the_values, the_delimiter )
                copy the_line to end of the_lines
            end tell
        end repeat
    end tell
        
    set the_lf to ASCII character 10
    set the_text to my join_items(the_lines, the_lf)
    my write_file(the_csvpath, the_text)

end export_csv


on run argv

    try
        set the_csvpath to (item 1 of argv)
        set the_sheetname to (item 2 of argv)
        set the_toprow  to (item 3 of argv)
        set the_delimiter to (item 4 of argv)
    on error
        set the_csvpath to "/Users/blyth/e/osx/numbers/out.csv"
        set the_sheetname to "Journal paper"
        set the_toprow to "3"
        set the_delimiter to "|"
    end try
    if (the_delimiter is "TAB") then
        set the_delimiter to tab
    end if
    my export_csv( the_csvpath, the_sheetname, the_toprow, the_delimiter )

end run




