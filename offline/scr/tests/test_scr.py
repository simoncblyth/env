"""
Scrape features to test

#. no-significant change, yields no update (except 1st run OR age_threshold)

How to test a scrape ? Granularity choice ?
#. full set of iterations  (simpler than iteration-by-iteration, which would entail interprocess communication) 

#. main process launches updater/scraper sub-processes (configured with clear time-to-live/max iterations)
   #. update simulation (with knows sequence of entries, designed to tickle features)
   #. scraper
#. main process blocks on joining those ... the examines tealeaves for expectations 

#. test specific parameters (for intervals/sleeps) in order for test to complete in reasonable time


"""
