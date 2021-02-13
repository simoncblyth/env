(*
/**
qt_trim
=========

Trims a movie using QuickTime Player

trimArg0
   when -ve is trimStart offset from total_duration, 
   when +ve is time from start 

trimArg1
   when -ve is trimEnd offset from total_duration
   when +ve is duration from trimStart

inPath
   input path 

outPath
   output path 

https://developer.apple.com/library/archive/technotes/tn2065/_index.html
**/
*)

on qt_trim( arg1, arg2, inPath, outPath )

    log "qt_trim : arg1: " & arg1 & " arg2: " & arg2 & " inPath: " & inPath & " outPath: " & outPath

    do shell script "touch \"" & outPath & "\""     -- without this cannot make the outFile alias
    set inFile to ((inPath as POSIX file) as alias)
    set outFile to ((outPath as POSIX file) as alias)

    tell application "QuickTime Player"
        --activate
        set doc to (open inFile)

        set total_duration to (duration of doc) 
        log "total_duration " & total_duration    

        -- -ve start seconds treated as relative to total_duration 
        if (arg1 is less than 0) then 
            set trimStart to arg1 + total_duration 
        else
            set trimStart to arg1 
        end if

        -- -ve or zero treased as relative to total_duration otherwise treated as a duration to be added to start
        if (arg2 is less than or equal to 0) then  
            set trimEnd to arg2 + total_duration   
        else
            set trimDuration to arg2
            set trimEnd to trimStart + trimDuration 
        end if
    
        if (trimEnd is greater than total_duration) then
            set trimEnd to total_duration
        end if

        log "trim from trimStart (" & trimStart & ") to trimEnd (" & trimEnd & ")   "

        trim front document from trimStart to trimEnd
        export front document in outFile using settings preset "1080p"
        close every document saving no
    end tell
end qt_trim

on run argv
    if count of argv is 4 then 
        set the_arg1 to (item 1 of argv)
        set the_arg2 to (item 2 of argv)
        set the_ipath to (item 3 of argv)
        set the_opath to (item 4 of argv)
    else
        set the_arg1 to 0
        set the_arg2 to 10 
        set the_ipath to "/Users/blyth/Movies/2018_10.mp4"
        set the_opath to "/Users/blyth/Movies/2018_10_testing.mp4"
    end if
    qt_trim(the_arg1, the_arg2, the_ipath, the_opath )
end run
