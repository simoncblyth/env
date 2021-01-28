(*
/**
qt_trim.applescript
=====================

File handling in applescipt is laborious and fragile, 

Contrast this applescript with the much cleaner bash 
approach in trim.sh trim.applescript


**/
*)

on get_stem_ext(the_name)
    set the_len to the count of characters in the_name
	set the_rname to (the reverse of every character of the_name) as text
	set the_dotpos to (offset of "." in the_rname) 
	set the_stem to text 1 thru -(1 + the_dotpos) of the_name
    set the_ext to text -(1 + the_dotpos) thru the_len of the_name
	return {the_stem, the_ext}
end get_stem

on get_dirname_basename(the_path)
	set the_path_len to the count of characters in the_path
	set the_rpath to (the reverse of every character of the_path) as text
	set the_slapos to (offset of "/" in the_rpath)
	set the_dirname to text 1 thru -(the_slapos + 1) of the_path
	set the_basename to text -(the_slapos - 1) thru the_path_len of the_path
	return {the_dirname, the_basename}
end get_dirname_basename


on qt_trim(the_trim_start, the_trim_duration, the_path)

    log "the_trim_start " & the_trim_start
    log "the_trim_duration " & the_trim_duration
    log "the_path " & the_path

    set {the_dirname, the_basename} to get_dirname_basename(the_path)
    log "the_dirname " & the_dirname
    log "the_basename " & the_basename

    --set the_name to name of (info for the_path)
    --log "the_name " & the_name

    set {the_stem, the_ext} to get_stem_ext(the_basename)
    log "the_stem " & the_stem 
    log "the_ext " & the_ext 

    --set the_ext to "mov"

    set the_trim_name to the_stem & "_qt_trim_" & the_trim_start & "_" & the_trim_duration & "." & the_ext
    log "the_trim_name " & the_trim_name 

    set the_trim_path to the_dirname & "/" & the_trim_name 
    log "the_trim_path " & the_trim_path 

    set the_file to the_path as POSIX file
    set the_alias to the_file as alias

    tell application "QuickTime Player"

        --activate 
        set doc to open the_alias
        set the_duration to duration of doc
        log "the_duration " & the_duration
   
        if (the_trim_start is greater than the_duration) then
            set the_trim_start to the_duration - the_trim_duration
            log "adjust the_trim_start " & the_trim_start
        end if

        set the_trim_end to the_trim_start + the_trim_duration
        if (the_trim_end is greater than the_duration) then
            set the_trim_end to the_duration
            log "adjusted the_trim_end " & the_trim_end
        else 
            log "the_trim_end " & the_trim_end
        end if

        trim doc from the_trim_start to the_trim_end

        log "exporting... to the_trim_path " & the_trim_path 
        set the_trim_file to file the_trim_path
      
        set the_save_path to a reference to POSIX file the_trim_path 
 
        tell document 1
            export in the_save_path 
            -- using settings preset "Computer"
        end tell 
 
    end tell
end qt_trim 




on run argv

    set the_trim_start_default to 0   -- seconds
    set the_trim_duration_default to 10  -- seconds
    set the_movie_path_default to "/Users/blyth/Movies/2018_10.mp4"

    try 
        set the_trim_start to (item 1 of argv)     
    on error
        set the_trim_start to the_trim_start_default
    end try 

    try 
        set the_trim_duration to (item 2 of argv)     
    on error
        set the_trim_duration to the_trim_duration_default
    end try 

    try 
        set the_movie_path to (item 3 of argv)
    on error
        set the_movie_path to the_movie_path_default 
    end try 


    qt_trim(the_trim_start, the_trim_duration, the_movie_path )

end run 


