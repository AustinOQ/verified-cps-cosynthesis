#!/bin/bash

set -eou pipefail

nuXmv -int SMV/mixing-model/model.smv <<EOF
go_msat
msat_check_invar_bmc -a een-sorensson -k 20
quit
EOF
