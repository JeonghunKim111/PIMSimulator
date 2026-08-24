# Required environment: NETLIST, TOP, SDC, LIB_MAX.
proc require_env {name} {
  if {![info exists ::env($name)] || $::env($name) eq ""} {
    puts stderr "error: required environment variable $name is not set"
    exit 2
  }
  return [file normalize $::env($name)]
}
set netlist [require_env NETLIST]
set sdc [require_env SDC]
if {![info exists ::env(TOP)] || $::env(TOP) eq ""} { puts stderr "error: TOP is not set"; exit 2 }
set liberty [require_env LIB_MAX]
foreach path [list $liberty $netlist $sdc] {
  if {![file isfile $path]} { puts stderr "error: input file not found: $path"; exit 2 }
}
read_liberty -max $liberty
read_verilog $netlist
link_design $::env(TOP)
read_sdc $sdc
check_setup
report_checks -path_delay max -group_path_count 10 -endpoint_path_count 10 -fields {slew cap input_pin}
report_worst_slack -max
report_tns -max
