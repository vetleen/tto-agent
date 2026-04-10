#!/bin/bash
# Add PulseAudio library path so ffprobe/ffmpeg can find libpulsecommon
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/app/.apt/usr/lib/x86_64-linux-gnu/pulseaudio"
