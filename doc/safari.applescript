tell application "Safari"
	set bounds of window 1 to {0, 22, 1024, 768 + 96 + 15}
	get bounds of window 1
	
	--   
	--   768 + 96 + 15 - 22 = 857
	--   857*2 = 1714
	--
	-- NB make sure there are already tabs present for consistency
	--
	-- use screen capture: shift-cmd-4 to get crosshairs with coordinates 
	-- window left corners
	--
	--       {0, 22}     window top left 
	--      {0, 96}    content top left       74 for Safari chrome including tabs
	--      {0, 863}   content bottom left
	--       {0, 878}   window bottom left 
	--
	--   screen captures:      2048 x 1714 
	--  after headtail crop (148, 30)        2048 x 1536    
	--         
	--  these are as expected 1714 - 148 - 30 = 1536
	--
	--       1024x768 *2 ->  2048x1536
	
end tell