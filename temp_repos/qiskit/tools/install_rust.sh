#!/bin/sh
if [ ! -d rust-installer ]; then
    mkdir rust-installer
    wget https://sh.rustup.rs -O rust-installer/rustup.sh
    sh rust-installer/rustup.sh -y -c llvm-tools
fi
. "$HOME/.cargo/env"
