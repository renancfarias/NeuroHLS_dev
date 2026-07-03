set ROOT [file normalize [file dirname [info script]]]
set COMPONENT_DIR [file join $ROOT event_driven_scnn]
set HLS_DIR [file join $COMPONENT_DIR hls]
set EXPORT_ZIP [file join $HLS_DIR impl export.zip]

# We are replacing this script's usage in the main script with vitis-run commands directly, 
# but keep it here empty or commented just in case.
exit
