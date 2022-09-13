mkdir SIMMC2_data
cd SIMMC2_data
wget https://github.com/facebookresearch/simmc2/raw/main/data/simmc2_scene_jsons_dstc10_public.zip
wget https://github.com/facebookresearch/simmc2/raw/main/data/simmc2_scene_jsons_dstc10_teststd.zip
wget https://github.com/facebookresearch/simmc2/raw/main/data/simmc2_scene_images_dstc10_teststd.zip
wget https://github.com/facebookresearch/simmc2/raw/main/data/simmc2_scene_images_dstc10_public_part1.zip
wget https://github.com/facebookresearch/simmc2/raw/main/data/simmc2_scene_images_dstc10_public_part2.zip
wget https://github.com/facebookresearch/simmc2/raw/main/data/fashion_prefab_metadata_all.json
wget https://github.com/facebookresearch/simmc2/raw/main/data/furniture_prefab_metadata_all.json
wget https://github.com/facebookresearch/simmc2/raw/main/dstc10/data/simmc2_dials_dstc10_train.json
wget https://github.com/facebookresearch/simmc2/raw/main/dstc10/data/simmc2_dials_dstc10_dev.json
wget https://github.com/facebookresearch/simmc2/raw/main/dstc10/data/simmc2_dials_dstc10_devtest.json
wget https://github.com/facebookresearch/simmc2/raw/main/dstc10/data/simmc2_dials_dstc10_teststd_public.json

unzip simmc2_scene_jsons_dstc10_public.zip
unzip simmc2_scene_jsons_dstc10_teststd.zip
unzip simmc2_scene_images_dstc10_teststd.zip
unzip simmc2_scene_images_dstc10_public_part1.zip
unzip simmc2_scene_images_dstc10_public_part2.zip

mv simmc2_scene_images_dstc10_public_part1/* simmc2_scene_images_dstc10_public_part2/
rm -r simmc2_scene_images_dstc10_public_part1
mv simmc2_scene_images_dstc10_public_part2 simmc2_scene_images_dstc10_public
mv simmc2_scene_jsons_dstc10_teststd/* public/
mv simmc2_scene_images_dstc10_teststd/* simmc2_scene_images_dstc10_public/

rm *zip
cd ../
mkdir TrainedModels
mkdir TrainedModels/pretrained_model
cd TrainedModels/pretrained_model
gdown 14VceZht89V5i54-_xWiw58Rosa5NDL2H
gdown 1qJYtsGw1SfAyvknDZeRBnp2cF4VNjiDE

