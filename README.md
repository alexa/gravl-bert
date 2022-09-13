# GraVL-BERT

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

## Download data
1. Run simmc2_download.sh and this creates a folder SIMMC2_data with all SIMMC2.0 data

## Create environment
works on cuda>11.0

```
cd GraVL-BERT
conda create -n gravlbert python=3.8
conda activate gravlbert
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install Cython
pip install -r requirements.txt
bash ./scripts/init.sh
```

## Generate labels from their original data files:

1. The folder Captioning contains everything to generate brief captions for non-target objects. Refer to the README file in it to train and generate captions.
```
	python test.py -model alexnet -epoch 71 -gpu_device 0
```
	raw_captions.json will be generated in SIMMC2_data folder. It contains captions of non-target objects in SIMMC2 data.

2. Run SIMMC_create_labels.py for train, dev and devtest. 
```
	python SIMMC_create_labels.py --mode train
	python SIMMC_create_labels.py --mode dev
	python SIMMC_create_labels.py --mode devtest
```
	This does three things
	(1) Convert metadata to text format and save as 'meta_text_all.json'
	(2) Generate a scene graph for each scene.json file and save as graph.npy 
	(3) Generate the processed label file used for training and testing. Files saved in SIMMC2_data/data_processed

## To train a model:
1. All necessary configurations are in cfgs/simmc/base_qa2r_4x16G_fp32.yaml (the filename does not have any meanings)
2. In GraVLBERT folder, run 
```
bash ./scripts/dist_run_single.sh 4 simmc2/train_end2end.py <config path> <model save path>
```
to train a file
for example: 
```
bash ./scripts/dist_run_single.sh 4 simmc2/train_end2end.py ./cfgs/simmc/base_qa2r_4x16G_fp32.yaml ../TrainedModels
```
## To evaluate a model:
1. If just want to see the results, in GraVLBERT folder, run 
	```
	python simmc2/train_end2end.py --cfg <config path> --model-dir <model save path> --evalonly True --savepath <prediction save path>
	```
	for example: 
	```
	python simmc2/train_end2end.py --cfg ./cfgs/simmc/base_qa2r_4x16G_fp32.yaml --model-dir '' --evalonly True --savepath eval_results.pkl
	```
2. To generate results in the DSTC required format. run 
	```
	python simmc2/test.py --cfg <config path> --ori-dialogue <original dialogue data provided by the official> --savepath <savepath>
	```
	for example: 
	```
	python simmc2/test.py --cfg ./cfgs/simmc/base_qa2r_4x16G_fp32.yaml --ori-dialogue ../SIMMC2_data/simmc2_dials_dstc10_teststd_public.json --savepath test_results.pkl
	```
	This script does three things:
		1. convert the raw data into the format our model uses. Save two temporary files, one for annotations and one for graph edges
		2. run model over the data, generate the results in DSTC required format. Two files are saved, one for output logits and the other for final outputs. The final outputs has 'dstc.json' surfix 
		3. run the offical evaluate function and return the metrics. If it is for the test data, this step returns error because there is no label in test data

## Files & Folders:
	GraVL-BERT/cfgs: 
		contains all config files
		the one we should use is cfgs/simmc/base_qa2r_4x16G_fp32.yaml (the file name does not have any meanings)
	GraVL-BERT/common:
		contains utility functions
	GraVL-BERT/external:
		BERT modules. Do not change
	GraVL-BERT/model:
		contains pretrained model downloaded from the VL-BERT GitHub
	GraVL-BERT/pretrain:
		scripts for pretraining
	GraVL-BERT/simmc2:
		the main file to run models on SIMMC2 dataset
	GraVL-BERT/scripts:
		contains the .sh files to run experiment
	GraVL-BERT/otherfiles:
		not used
