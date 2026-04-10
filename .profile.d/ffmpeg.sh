#!/bin/bash
# Add library subdirectories the apt buildpack doesn't put on LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/app/.apt/usr/lib/x86_64-linux-gnu/pulseaudio:/app/.apt/usr/lib/x86_64-linux-gnu/blas:/app/.apt/usr/lib/x86_64-linux-gnu/lapack"
