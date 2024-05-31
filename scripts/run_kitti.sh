DATASET=/media/data/KITTI/raw/sequences
SCENES=(00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21)
for scene in ${SCENES[@]}; do
    echo Running inference on $scene ...
    python demo.py -b 8 -i $DATASET/$scene/image_2/ -o $DATASET/$scene/depth/metric3d-vit_giant2/ --intrinsics=calib/kitti.txt
    echo Done.
done
