DATASET=/media/data/nerf/ours
SCENES=(kitchen floor)
for scene in ${SCENES[@]}; do
    echo Running inference on $scene ...
    python demo.py -b 8 -i $DATASET/$scene/raw/images/ -o $DATASET/$scene/raw/metric3d-vit_giant2/ 
    echo Done.
done
