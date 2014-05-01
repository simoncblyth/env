(*
Usage::

    osascript numbers_import_csv.applescript update.csv "Journal paper" 3
    osascript numbers_import_csv.applescript export.csv "Journal paper" 3
 
Reads from a csv file and inserts the values into additional rows 
of the named worksheet of the frontmost spreadsheet 
opened in OSX Numbers.app 

The third argument is the 1-based row number above which entries are inserted
one by one.  
   

PERHAPS:

#. reverse order to make csv order same as final spreadsheet ?

*)


on split_csvline(the_line, the_delimiter)

    set {tids, text item delimiters} to {text item delimiters, the_delimiter}
    set the_items to text items of the_line
    set text item delimiters to tids
    return the_items

end split_csv


on add_rows_( the_sheetname, the_items, the_toprow)

    tell application "Numbers" to tell document 1 to tell sheet the_sheetname to tell table 1
           set the_row to (add row above row the_toprow)
           repeat with the_index from 1 to  (count of the the_items)
               set the_value to item the_index of the_items
               if (the_value is not missing value and the_value is not "missing value") then
                   tell cell the_index of the_row 
                       set format to text
                       set value to the_value as text
                   end tell                 
               end if 
           end repeat  -- over columns of the csv
    end tell

end add_rows_


on add_rows(the_csvpath, the_sheetname, the_toprow, the_delimiter, the_reverse )

    set the_lf to ASCII character 10
    set the_lines to (read the_csvpath using delimiter the_lf )

    if (the_reverse is true) then
        set the_lines_used to reverse of the_lines
    else
        set the_lines_used to the_lines
    end if

    repeat with the_line_index from 1 to (count of the_lines)

       set the_line to item the_line_index of the_lines_used
       set the_items to split_csvline(the_line, the_delimiter) 
       add_rows_( the_sheetname, the_items, the_toprow )

    end repeat       -- over lines of the csv file        
end add_rows


on run argv

    try
        set the_csvpath to (item 1 of argv)
        set the_sheetname to (item 2 of argv)
        set the_toprow  to (item 3 of argv)
        set the_delimiter to (item 4 of argv)
    on error
        set the_csvpath to "/Users/blyth/e/osx/numbers/demo.csv"
        set the_sheetname to "Journal paper"
        set the_toprow to "3"
        set the_delimiter to "|"
    end try
    if (the_delimiter is "TAB") then
        set the_delimiter to tab
    end if
    set the_reverse to true
    add_rows( the_csvpath, the_sheetname, the_toprow, the_delimiter, the_reverse )

end run




