This code is from [yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)

Download their trained model and finetune on SIMMC2 https://drive.google.com/open?id=1xF8dfIDsz57ZrX7bKApOakyjm1GoelJm


To train the model:
```
python train.py -model alexnet -epoch 100 -gpu_device 0
	models are saved with epoch name, so the loaded model will follow the epoch argument
```

To test the model:
```
python test.py -model alexnet -epoch 71 -gpu_device 0
```
	same as train.py