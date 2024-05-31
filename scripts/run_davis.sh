DATASET=/media/data/DAVIS
SCENES=(camel car-turn parkour snowboard stroller stunt swing tennis)
for scene in ${SCENES[@]}; do
    echo Running inference on $scene ...
    python demo.py -b 8 -i $DATASET/JPEGImages/Full-Resolution/$scene/ -o $DATASET/Depth/Full-Resolution/metric3d-vit_giant2/$scene/
    echo Done.
done
