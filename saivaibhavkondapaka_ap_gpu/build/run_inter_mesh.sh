#!/usr/bin/env bash
echo -e "MeshA\tMeshB\tFacesA\tFacesB\tPairs\tTime_ms\tMpairs_s\tIntersected" \
  > inter_mesh_results.tsv

BATCH=10000000

for A in ../meshes/*.off; do
  facesA=$(awk 'NR==2{print $2; exit}' "$A")
  for B in ../meshes/*.off; do
    facesB=$(awk 'NR==2{print $2; exit}' "$B")
    pairs=$((facesA*facesB))
    printf "%s\t%s\t%d\t%d\t%d\t" \
      "$(basename "$A")" "$(basename "$B")" \
      "$facesA" "$facesB" "$pairs" \
      >> inter_mesh_results.tsv

    ./sai_tri_gpu "$A" "$B" "$BATCH" \
      >> inter_mesh_results.tsv
  done
done
