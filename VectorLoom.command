#!/bin/zsh
set -e
cd "${0:A:h}"
(sleep 1 && open http://127.0.0.1:3000) &
exec cargo run --release
