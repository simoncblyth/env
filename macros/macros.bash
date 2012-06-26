macros-usage(){ cat << EOU

EOU
}


macros-gui(){

echo ==== macros-gui writing $DYM/gui.mac ===

cat << EOM > $DYM/gui.mac
#
# dont edit this file, instead edit the generator env:/trunk/macros/macros.bash 
#
#   invoke this with:
#        /control/execute $DYM/gui.mac 
#
#   ... curiously the last seems not to appear, so duplicate it
#
#
#   ... Actually would be better for the derived file not to be kept in repository folders, otherwise 
#    have non-clean revision for no good reason 
#

/gui/addMenu "macros" "macros" 
/gui/addButton "macros" "set-scene"        "/control/execute $DYM/set-scene.mac" 
/gui/addButton "macros" "add-trajectories" "/control/execute $DYM/add-trajectories.mac" 
/gui/addButton "macros" "scb-traj"         "/control/execute $DYM/scb-traj.mac" 
/gui/addButton "macros" "next-events"       "/control/execute $DYM/next-event.mac" 
/gui/addButton "macros" "color-particles"  "/control/execute $DYM/scb-draw-by-particle-id.mac"
/gui/addButton "macros" "vis-geometry-list"  "/vis/geometry/list"
/gui/addButton "macros" "vis-geometry-list"  "/vis/geometry/list"

EOM


}
