puts "COMMANDS=[info commands *saif*]"
foreach c {open_saif close_saif log_saif report_saif write_saif} {
  puts "## $c"
  if {[catch {help $c} msg]} {
    puts "ERR=$msg"
  } else {
    puts $msg
  }
}
exit
