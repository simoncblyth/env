--
--
--    1:1   1.0 
--    5:4   1.25    
--    4:3   1.333     Standard 4:3: 320x240, 640x480, 800x600, 1024x768
--    8:5   1.6 
--    16:9  1.777     Widescreen 16:9: 640x360, 800x450, 960x540, 1024x576, 1280x720, and 1920x1080
--
--
--   Doubling done by retina screen means cannot directly fit 1920x1080
--   so use same aspect 1280x720
--   or half it to 960x540 ? 
--  

tell application "Safari"

    --set width to 1024
    --set height to 768

    set width to  1280
    set height to  720

    -- half target is too small to work on 
    --set width to  960
    --set height to  540



    -- In [9]: (1920./120.)/(1080./120.)
    -- Out[9]: 1.7777777777777777
    -- In [10]: 16./9.
    -- Out[10]: 1.7777777777777777

	set bounds of window 1 to {0, 22, width, height + 96 + 15}
	get bounds of window 1

    --   1280x720
    --
    --    {0, 22}   window top left 
    --    {0, 96}   content top left
    --    {0, 815}  content bottom left  815-96 = 719  
    --    {0, 831}  window bottom left 
    --    {1280,22} window top right   
    --
    --
    --    1024x768	
	--   
	--   768 + 96 + 15 - 22 = 857
	--   857*2 = 1714
	--
	-- NB make sure there are already tabs present for consistency
	--
	-- use screen capture: shift-cmd-4 to get crosshairs with coordinates 
	-- window left corners
	--
	--      {0, 22}    window top left 
	--      {0, 96}    content top left    96-22=74 for Safari chrome including tabs
	--      {0, 863}   content bottom left
	--      {0, 878}   window bottom left 
	--
	--   screen captures:      2048 x 1714 
	--  after headtail crop (148, 30)        2048 x 1536    
	--         
	--  these are as expected 1714 - 148 - 30 = 1536
	--
	--       1024x768 *2 ->  2048x1536
	
end tell
