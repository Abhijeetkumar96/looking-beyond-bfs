#!/usr/bin/env bash

DATASET_DIR="/home/abhijeet/datasets/medium_datasets/ecl_graphs"

for g in \
  web-BerkStan \
  as-Skitter \
  higgs-twitter \
  coPapersDBLP \
  sx-stackoverflow \
  road_usa \
  soc-LiveJournal1 \
  kron_g500-logn20 \
  europe_osm \
  kron_g500-logn21 \
  com-Orkut \
  uk-2002
do
  echo "=============================="
  echo "Running dataset: $g"
  echo "=============================="
  ./main "$DATASET_DIR/$g.egr"
done
