set argv [list "vitis_proj" "sol"]
set argc 2
open_project "vitis_proj"
open_solution "sol"
cosim_design -setup
exit
