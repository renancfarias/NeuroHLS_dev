
# Runs the C-simulation

#set_part {xc7z020clg400-1}

# Arguments (in order):
#   - Project's name
#   - Solutions's name
#   - Clock's max period in ns
#   - The used FPGA's part

set PROJ [lindex $argv 0]
set SOL  [lindex $argv 1]
set CLK  [lindex $argv 2]
set PART [lindex $argv 3]

open_project $PROJ

set_top snn_to_hls

open_solution $SOL

create_clock -period $CLK -name default_clk

set_part $PART

csynth_design

exit