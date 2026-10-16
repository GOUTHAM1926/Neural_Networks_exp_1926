#!/bin/bash
# usage: dump.sh <obj> <name-substring> <out.sass>: SASS of the ONE kernel whose mangled name contains the substring,
# instructions only (addresses and encodings stripped); <out.sass>.norm also has constant-bank offsets masked.
fn=$(cuobjdump -symbols "$1" 2>/dev/null | grep -o '_Z[A-Za-z0-9_]*' | grep -- "$2" | sort -u)
n=$(echo "$fn" | grep -c .)
[ "$n" != 1 ] && { echo "dump.sh: $n functions match $2"; echo "$fn"; exit 1; }
cuobjdump -sass -fun "$fn" "$1" | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#^\s+/\*[0-9a-f]+\*/\s+##; s#\s*/\*[^*]*\*/\s*$##; s#\s+;#;#' > "$3"
sed -E 's/c\[0x0\]\[0x[0-9a-f]+\]/c[0x0][X]/g' "$3" > "$3.norm"
echo "$(wc -l < "$3") instructions  $fn"
