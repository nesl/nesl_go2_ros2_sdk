#!/usr/bin/env bash
# The lab map, built by the console from the dog's lidar (go2_mapper.py). Same as the Map panel buttons.
#   ./go2_map.sh start    start mapping (adds to the saved map when lined up with it, else a new map)
#   ./go2_map.sh new      start a new map (replaces the saved one when you save)
#   ./go2_map.sh clear    erase the map on screen (the saved file is kept until you save)
#   ./go2_map.sh extend   keep adding to the saved map (once lined up with it)
#   ./go2_map.sh save     stop mapping and save to ~/oliver/go2_maps/lab.* (the old one goes to history/)
#   ./go2_map.sh stop     stop mapping without saving
#   ./go2_map.sh revert   drop unsaved changes: show the saved map again
#   ./go2_map.sh align    line the dog up with the saved map again
#   ./go2_map.sh status
#   ./go2_map.sh history              list earlier saved versions
#   ./go2_map.sh restore <YYYYMMDD_HHMMSS>   make an earlier version the saved map again and show it
#                                     (the current saved map goes to history/ first)
MAPS=$(dirname "$(readlink -f "$0")")/go2_maps
case "${1:-}" in
start|new|extend|save|stop|clear|align|revert)
    curl -s -H 'Content-Type: application/json' -d "{\"action\":\"$1\"}" http://localhost:8080/map; echo ;;
status)
    curl -s http://localhost:8080/state.json | python3 -c '
import json, sys
d = json.load(sys.stdin); m = d["map"]
print("no map" if not m else "%s: %s, %s m2 seen, saved %s, robot at %s" % (m["mode"], m["status"], m["mapped_m2"], m.get("saved_at"), d["map_pose"]))' ;;
history)
    echo "saved now: $(date -r "$MAPS/lab.grid.npz" '+%Y-%m-%d %H:%M:%S')"
    ls "$MAPS/history" | sed -n 's/^lab_\([0-9_]*\)\.grid\.npz$/  \1/p' ;;
restore)
    v=${2:-}; src="$MAPS/history/lab_$v"
    [ -n "$v" ] && [ -f "$src.grid.npz" ] || { echo "no version '$v'; see: $0 history"; exit 1; }
    now=$(date -r "$MAPS/lab.grid.npz" '+%Y%m%d_%H%M%S')
    for e in grid.npz pgm yaml; do
        [ -f "$MAPS/history/lab_$now.$e" ] || cp -p "$MAPS/lab.$e" "$MAPS/history/lab_$now.$e"
        cp -p "$src.$e" "$MAPS/lab.$e"
    done
    echo "restored $v (the map saved at $now is in history/)"
    curl -s -H 'Content-Type: application/json' -d '{"action":"revert"}' http://localhost:8080/map >/dev/null && echo "console now shows it" ;;
*)
    sed -n '2,14p' "$0"; exit 1 ;;
esac
