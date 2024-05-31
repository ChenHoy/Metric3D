DATASET=/media/data/totalrecon
# SCENES=(cat1-stereo000 cat1-stereo001 cat2-stereo000 cat2-stereo001 cat3-stereo000 dog1-stereo000 dog1-stereo001 human1-stereo000 human2-stereo000 humancat-animal-stereo000 humancat-human-stereo000 humandog-animal-stereo000 humandog-human-stereo000)
SCENES=(humancat-animal-stereo000 humancat-human-stereo000 humandog-animal-stereo000 humandog-human-stereo000)
for scene in ${SCENES[@]}; do
    echo Running inference on $scene ...
    python demo.py -i $DATASET/$scene-leftcam/images/ -o $DATASET/$scene-leftcam/metric3d-vit_giant2/ --intrinsics=calib/totalrecon/$scene.txt -b 8
    echo Done.
done
