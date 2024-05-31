DATASET=/media/data/Replica
SCENES=(office0 office1 office2 office3 office4 room0 room1 room2)
for scene in ${SCENES[@]}; do
    echo Running inference on $scene ...
    python demo.py -b 8 -i $DATASET/$scene/results/ -o $DATASET/$scene/metric3d-vit_giant2/ --intrinsics=calib/replica.txt
    echo Done.
done
