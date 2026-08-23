
# Runs the C-simulation

# Arguments (in order):
#   - Project's name
#   - Solutions's name

set PROJ [lindex $argv 0]
set SOL  [lindex $argv 1]

open_project $PROJ
open_solution $SOL

csim_design

exit