# RTL result retention policy

This directory versions only the compact evidence needed to audit a reported
experiment: summary Markdown/CSV, Yosys `stat` reports, runtimes, and source/tool/
library hashes.

Mapped netlists, resolved Yosys scripts, synthesis logs, and timing-sweep work
directories are reproducible generated products and are ignored. Recreate them
with the scripts under `../scripts/` and a legally obtained Liberty file. Never
commit a third-party Liberty, LEF, DB, or GDS file unless its redistribution
license has been reviewed explicitly.
