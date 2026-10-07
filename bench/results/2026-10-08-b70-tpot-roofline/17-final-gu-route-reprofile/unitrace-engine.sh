#!/usr/bin/env bash
exec /tools/unitrace/bin/unitrace --system-time --chrome-device-logging --chrome-event-buffer-size 100000 --include-kernels native_gu_ --output-dir-path /artifacts/tpot-20261007-r81j21y4/17-final-gu-route-reprofile --teardown-on-signal 15 /artifacts/tpot-20261007-r81j21y4/bin/final/strata "$@"
