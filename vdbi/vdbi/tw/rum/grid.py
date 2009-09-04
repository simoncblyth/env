"""
   Stomps on the edit and delete actions ...
   avoiding the icons/links for each row of the table
"""

from tw.rum import RumDataGrid
RumDataGrid.actions = ['show']    
dummy = 42


