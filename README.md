# Multiple Style Combine & Transfer - Image / Video
It is based on the pytorch [fast-neural-style](https://github.com/pytorch/examples/tree/master/fast_neural_style) example for artistic style transfer. We further modify the instance normalization in the transform network, making it conditional with multiple style images. So, you can combine effects from multiple style images and transfer to a target image. We also use this model to transfer videos in the multiple-style-combined ways. But note that the video style transfer is not real time. The video is pre-processed frame-by-frame and converted back to video after style transfer.

### Training content image datasets
[COCO 2014 Training images dataset [83K/13GB]](http://images.cocodataset.org/zips/train2014.zip)


## Usage

Train
```
python neural_style.py train --dataset /scratch/ywn202/style_train_data --style-image ./images/style_images/ --save-model-dir ./trained_models --epochs 5 --cuda 1 --log-interval 100 --batch-size 12
```
* `--dataset`: the content images that used to train the tranformer network in fast-neural-style-transfer.
* `--style-image`: the dir where the style images stored. Images should be in .jpg format.
* `--batch-size`: number of images fed in each batch does not need to be the same to the number of style images.

Stylize Image
```
python neural_style.py eval --content-image ./images/content_images/hkbuilding.jpg --model ./trained_models/epoch_5_Sat_Dec__1_102704_2018_100000_10000000000.model --output-image hkbuilding --cuda 1 --style-num 4 --style-control 0 0 1 0
```
* `--content-image`: the image to stylized.
* `--model`: the trained model with the transformer network to stylize the image.
* `--output-image`: 
* `--style-num`: total number of style images, must be the same as the amount used in training.
* `--style-control`: a vector to specify which style(s) is/are tranferred. 

Stylize Video
```

```
* `--style-num`: total number of style images, must be the same as the amount used in training.
* `--style-control`: a vector to specify which style(s) is/are tranferred. 


## Results

### Pretrained model
Models trained by us are saved in the **trained_models** folder. It is trained with the style images stored in the /images/style_images

Example of combining 4 different styles:
<div align='center'>
  <img src='images/content_images/hkbuilding.jpg' height="200px">		
</div>

<p>
<img src="images/output_images/hkbuilding_styles_combined.jpg" width="1000" height="550" />
</p>
