
# This TCL file resets the project and adds the files
# to enable the simulation and synthesis

# Arguments (in order):
#   - Project's name

set PROJ [lindex $argv 0]
open_project -reset $PROJ

# ---------------------------------------------
# Adding SNN implementation files
# ---------------------------------------------

add_files snn_implementation.cpp
add_files snn_implementation.h
add_files weights.h
add_files types_and_params.h

# ---------------------------------------------
# Adding testbench data
# ---------------------------------------------

add_files tb_data/data.txt
add_files tb_data/targets.txt

# ---------------------------------------------
# Adding NeuroHls files
# ---------------------------------------------

add_files neuro_hls_functions/bit_type.h
add_files neuro_hls_functions/dense.h

# ---------------------------------------------
# Adding Testbench file
# ---------------------------------------------

add_files -tb testbench.cpp

exit