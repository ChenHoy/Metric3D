DATASET=/media/data/TartanAir
SCENES=(abandonedfactory japanesealley seasidetown westerndesert)
DIFFICULTY=(Easy Hard)
SEQUENCES=(P000 P001 P002 P004 P005 P006 P008 P009 P010 P011)
for scene in ${SCENES[@]}; do
    echo Difficulty: $diff
    for diff in ${DIFFICULTY[@]}; do
        for sequence in ${SEQUENCES[@]}; do
            echo Running inference on $scene/$diff/$sequence/ ...
            python demo.py -b 8 -i $DATASET/$scene/$diff/$sequence/image_left/ -o $DATASET/$scene/$diff/$sequence/metric3d-vit_giant2/ --intrinsics=calib/tartanair.txt
            echo $sequence Done.
        done
        echo $diff Done.
    done
    echo $scene Done.
done
