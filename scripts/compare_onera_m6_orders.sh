#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SU2_MESH="$APP_DIR/public-data/onera_m6/mesh_ONERAM6_inv_ffd.su2"
OP2_MESH="$APP_DIR/meshes-op2/onera_m6.h5"
FIRST_CFG="$APP_DIR/configs/onera_m6.cfg"
SECOND_CFG="$APP_DIR/configs/onera_m6_second_order.cfg"

RHO_INF="1.2250122659906946"
U_INF="285.26805348028574"
V_INF="0.0"
W_INF="15.249834230540356"
P_INF="101325.0"
WALL_GROUPS="1,2,3"

FIRST_SOLUTION="$APP_DIR/outputs-op2/onera_m6_solution.h5"
SECOND_SOLUTION="$APP_DIR/outputs-op2/onera_m6_second_order_solution.h5"
FIRST_SUMMARY="$APP_DIR/outputs-op2/onera_m6_summary.json"
SECOND_SUMMARY="$APP_DIR/outputs-op2/onera_m6_second_order_summary.json"
FIRST_CP="$APP_DIR/outputs-op2/onera_m6_surface_cp.csv"
SECOND_CP="$APP_DIR/outputs-op2/onera_m6_second_order_surface_cp.csv"
COMPARISON_JSON="$APP_DIR/outputs-op2/onera_m6_order_comparison.json"

"$APP_DIR/scripts/download_onera_m6_su2_case.sh"

if [ ! -f "$OP2_MESH" ]; then
  "$APP_DIR/scripts/preprocess_mesh.sh" onera_m6 "$OP2_MESH" "$SU2_MESH"
fi

make -C "$APP_DIR" seq
"$APP_DIR/scripts/build_helpers.sh"

"$APP_DIR/nssolver_op2_seq" --config "$FIRST_CFG"
"$APP_DIR/.helpers/bin/nssolver_onera_m6_validation_helper" \
  "$OP2_MESH" "$FIRST_SOLUTION" "$FIRST_SUMMARY" "$FIRST_CP" \
  "$RHO_INF" "$U_INF" "$V_INF" "$W_INF" "$P_INF" "$WALL_GROUPS"

"$APP_DIR/nssolver_op2_seq" --config "$SECOND_CFG"
"$APP_DIR/.helpers/bin/nssolver_onera_m6_validation_helper" \
  "$OP2_MESH" "$SECOND_SOLUTION" "$SECOND_SUMMARY" "$SECOND_CP" \
  "$RHO_INF" "$U_INF" "$V_INF" "$W_INF" "$P_INF" "$WALL_GROUPS"

python3 - <<'PY' "$FIRST_SUMMARY" "$SECOND_SUMMARY" "$FIRST_CP" "$SECOND_CP" "$COMPARISON_JSON"
import csv
import json
import sys

first_summary_path, second_summary_path, first_cp_path, second_cp_path, out_path = sys.argv[1:]

with open(first_summary_path) as f:
    first = json.load(f)
with open(second_summary_path) as f:
    second = json.load(f)

def cp_stats(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cps = [float(r["cp"]) for r in rows]
    upper = [float(r["cp"]) for r in rows if int(r["group_id"]) == 3]
    return {
        "count": len(cps),
        "cp_min": min(cps),
        "cp_max": max(cps),
        "cp_avg": sum(cps) / len(cps),
        "upper_cp_min": min(upper),
        "upper_cp_max": max(upper),
        "upper_cp_avg": sum(upper) / len(upper),
    }

comparison = {
    "first_order": {
        "summary": first,
        "cp_stats": cp_stats(first_cp_path),
    },
    "second_order": {
        "summary": second,
        "cp_stats": cp_stats(second_cp_path),
    },
    "delta_second_minus_first": {
        "cd": second["coefficients"]["cd"] - first["coefficients"]["cd"],
        "cl": second["coefficients"]["cl"] - first["coefficients"]["cl"],
        "cx": second["coefficients"]["cx"] - first["coefficients"]["cx"],
        "cz": second["coefficients"]["cz"] - first["coefficients"]["cz"],
    },
}

with open(out_path, "w") as f:
    json.dump(comparison, f, indent=2)

print(json.dumps(comparison["delta_second_minus_first"], indent=2))
PY

echo "comparison summary: $COMPARISON_JSON"
