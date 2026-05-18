#!/bin/bash
# Records audio from Spotify (PipeWire-pulse or PulseAudio) and splits it
# into per-track tagged Ogg Vorbis files using MPRIS metadata via playerctl.

set -u

script_dir=$(dirname "$(readlink -f "$0")")

encoder=ogg
ext=ogg
musicdir=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --mp3) encoder=mp3; ext=mp3 ;;
    -h|--help) echo "Usage: $0 [--mp3] [music_directory]" >&2; exit 0 ;;
    *) musicdir=$1 ;;
  esac
  shift
done
musicdir=${musicdir:-.}

deps=(pactl parec playerctl wget awk)
if [[ $encoder == mp3 ]]; then
  deps+=(lame id3v2)
else
  deps+=(oggenc vorbiscomment)
fi
for cmd in "${deps[@]}"; do
  command -v "$cmd" >/dev/null || { echo "missing dependency: $cmd" >&2; exit 1; }
done

# Find the Spotify sink-input. Under PipeWire, sink-inputs themselves carry
# only generic node names (e.g. "audio-src"); the application identity lives
# on the owning client. So: first collect client IDs whose application.name
# or application.process.binary is "spotify", then find a sink-input owned
# by one of them.
spotify_clients=$(pactl list clients | awk '
  /^Client #/ {
    if (cid != "" && hit) print cid
    sub("#","",$2); cid=$2; hit=0
  }
  /application\.(name|process\.binary)/ && tolower($0) ~ /"spotify"/ { hit=1 }
  END { if (cid != "" && hit) print cid }
')

spotify=$(pactl list sink-inputs | awk -v ids="$spotify_clients" '
  BEGIN { n = split(ids, arr, "\n"); for (i = 1; i <= n; i++) m[arr[i]] = 1 }
  /^Sink Input #/ { sub("#","",$3); idx = $3 }
  /^[[:space:]]+Client:[[:space:]]/ { if ($2 in m) { print idx; exit } }
')

if [[ -z $spotify ]]; then
  echo "Spotify is not running (no sink-input found). Play a song first." >&2
  exit 1
fi

# Create our recording sink if not already present
if ! pactl list short sinks | awk '{print $2}' | grep -qx spotify; then
  pactl load-module module-null-sink \
    sink_name=spotify sink_properties=device.description=SpotifyRipper >/dev/null
fi

default_sink=$(pactl get-default-sink)
rec_pid=""
loopback_module=""

cleanup() {
  [[ -n $rec_pid ]] && pkill -P "$rec_pid" 2>/dev/null
  [[ -n $loopback_module ]] && pactl unload-module "$loopback_module" 2>/dev/null
  pactl move-sink-input "$spotify" "$default_sink" 2>/dev/null
}
trap cleanup EXIT INT TERM

# Move Spotify into our isolated sink so its audio is captured cleanly
pactl move-sink-input "$spotify" spotify

# Loop the captured audio back to the default sink in real time so the user
# can still hear what's being recorded. Low latency keeps it close to live.
loopback_module=$(pactl load-module module-loopback \
  source=spotify.monitor sink="$default_sink" latency_msec=50)

artist=""; album=""; title=""; tracknumber=""

stop_recording() {
  [[ -z $rec_pid ]] && return
  # rec_pid is a subshell wrapping the pipeline; pkill -P reaches both
  # parec and oggenc as its direct children. Killing only the subshell
  # would orphan them.
  pkill -P "$rec_pid" 2>/dev/null
  wait "$rec_pid" 2>/dev/null
  rec_pid=""
}

while read -r line; do
  if [[ $line == "__SWITCH__" ]]; then
    stop_recording

    tmpfile="tmp.$ext"
    if [[ -n $title ]] && [[ -s $tmpfile ]]; then
      case $encoder in
        mp3)
          id3v2 -a "$artist" -A "$album" -t "$title" -T "$tracknumber" "$tmpfile" >/dev/null
          ;;
        *)
          vorbiscomment -a "$tmpfile" \
            -t "ARTIST=$artist" \
            -t "ALBUM=$album" \
            -t "TITLE=$title" \
            -t "TRACKNUMBER=$tracknumber"
          ;;
      esac

      saveto="$musicdir/${artist//\//_}/${album//\//_}"
      mkdir -p "$saveto"
      prefix=""
      [[ $tracknumber =~ ^[0-9]+$ ]] && prefix=$(printf '%02d ' "$tracknumber")
      target="$saveto/${prefix}${title//\//_}.$ext"
      mv "$tmpfile" "$target"
      echo "Saved: $target"

      if [[ -s cover.jpg && ! -e "$saveto/cover.jpg" ]]; then
        mv cover.jpg "$saveto/cover.jpg"
      fi
      rm -f cover.jpg
    fi

    artist=""; album=""; title=""; tracknumber=""

    echo "RECORDING"
    case $encoder in
      mp3)
        ( parec -d spotify.monitor |
          lame -r -s 44.1 --bitwidth 16 --signed --little-endian -m s -b 192 - tmp.mp3
        ) 2>/dev/null &
        ;;
      *)
        ( parec -d spotify.monitor | oggenc -b 192 -o tmp.ogg --raw - ) 2>/dev/null &
        ;;
    esac
    rec_pid=$!
  else
    key=${line%%=*}
    val=${line#*=}
    case $key in
      artist)      artist=$val;      echo "Artist = $val" ;;
      album)       album=$val;       echo "Album  = $val" ;;
      title)       title=$val;       echo "Title  = $val" ;;
      tracknumber) tracknumber=$val; echo "Track# = $val" ;;
      arturl)
        if [[ -n $val ]]; then
          wget -q -O cover.jpg "$val" || rm -f cover.jpg
        fi
        ;;
    esac
  fi
done < <("$script_dir/notify.sh")
