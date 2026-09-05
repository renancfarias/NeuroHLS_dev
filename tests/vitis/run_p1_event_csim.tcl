open_project -reset /tmp/neurohls_p1_event_csim
set_top p1_event_top
add_files tests/vitis/p1_event_top.cpp
add_files -tb tests/vitis/p1_event_tb.cpp
open_solution -reset sol -flow_target vitis
set_part {xcu250-figd2104-2L-e}
create_clock -period 10
csim_design
exit
