# Visual Genome Feature Extraction for VQA task

Generation output features corresponding to salient image regions.
These bottom-up attention features can typically be used as a drop-in replacement for CNN features in attention-based image captioning 
and visual question answering (VQA) models.

Feature extraction approach based on Peter Anderson et al [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998) (source code is available in [repo](https://github.com/peteanderson80/bottom-up-attention)).
Provided code show how to train large-scale instance segmentation model on Visual Genome dataset and extract features from trained model.
The training procedure based on [R. Hu, P. Dollár, K. He, T. Darrell, R. Girshick, Learning to Segment Every Thing](https://arxiv.org/pdf/1711.10370.pdf) paper.

## Dataset preparation
1. Download Visual Genome dataset, 1.2 version from [website](http://visualgenome.org/api/v0/api_home.html)
required: 
  * images [part 1 (9.2 GB)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part 2 (5.47 GB)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip)
  * [image meta data (17.62 MB)](http://visualgenome.org/static/data/dataset/image_data.json.zip)
  * [scene graph representation (739.37 MB)](http://visualgenome.org/static/data/dataset/scene_graphs.json.zip)
  and unpack downloaded archives to `data/vg` directory.
2. Download [objects vocabulary](https://github.com/peteanderson80/bottom-up-attention/blob/master/data/genome/1600-400-20/objects_vocab.txt) file with 1600 cleaned objects
and place it in `data/vg` directory
3. Download [train](https://github.com/peteanderson80/bottom-up-attention/blob/master/data/genome/train.txt)/[val](https://github.com/peteanderson80/bottom-up-attention/blob/master/data/genome/val.txt)/[test](https://github.com/peteanderson80/bottom-up-attention/blob/master/data/genome/test.txt) images splits and place them in `data/vg` directory
4. Convert original Visual Genome annotation to xml format, using following command:

```bash
python tools/visual_genome_to_xml.py --objects_vocab_file data/vg/objects_vocab.txt
```
5 Download [MSCOCO](http://cocodataset.org/#download) dataset (required for mask training)
required:
* [2017 Train images](http://images.cocodataset.org/zips/train2017.zip)
* [2017 Val images](http://images.cocodataset.org/zips/val2017.zip)
* [Train/Val annotations 2017](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

Finally, you should have such dataset structure:
````
data
|_vg
|  |_ VG_100K
|  |  |_ 2.jpg
|  |  |_ ...
|  |_ VG_100K_2
|  |_ xml
|  |    |_ 1.xml
|  |    |_ ...
|  |_ image_data.json
|  |_ scene_graphs.json
|  |_ objects_vocab.txt
|  |_ train.txt
|  |_ val.txt
|  |_ test.txt
|_ coco
|     |_ annotations
|     |            |_ instances_train2017.json
|     |            |_ instances_val2017.json
|     |_ train2017
|     |          |_ 000000000009.jpg
|     |          |_ ...
|     |_ val2017
...
````
## Training
The training procedure has step-wise strategy.
First in Stage 1, a Faster R-CNN detector is trained on all the 1600 Visual Genome classes.
Then in Stage 2, the mask branch (with the weight transfer function) is added and trained on the mask data of the 80 COCO classes. 
Finally, the mask branch is applied on all 1600 Visual Genome classes.

Before training on the mask data of the 80 COCO classes in Stage 2, a "surgery" is done to convert the 1600 VG detection weights to 80 COCO detection weights, so that the mask branch only predicts mask outputs of the 80 COCO classes (as the weight transfer function only takes as input 80 classes) to save GPU memory. 
After training, another "surgery" is done to convert the 80 COCO detection weights back to the 1600 VG detection weights.

Step-by-step training procedure:
1. Train Faster RCNN model:

```bash
python tools/train.py configs/visual_genome/faster_rcnn_r101_fpn_1x.py

```
2. Apply weights surgery procedure:

```bash
python tools/tensors_sergery/convert_vg_to_coco.py --input_model <saved_checkpoint_path> --output_model 1601_to81.pth

```
3. Train Mask RCNN model

```bash
python tools/train.py configs/visual_genome/mask_rcnn_r101_fpn_1x.py
```

4. Restore Faster RCNN weights:
```bash
python tools/tensors_sergery/restore_from_backup.py --input_model <saved_checkpoint_path> --output_model final_mask_rcnn_vg.pth
```

## Evaluate

For Faster RCNN and Mask RCNN validation suitable standart `tools/test.py` script.
If you want compare models prediction with gt, you are able to use `tools/eval_coco.py`, but you need to convert Visual Genome dataset to MSCOCO annotation format
(following script can be found in `tools/dataset_conversion/visual\_genome_to_coco.py`)
**Note:** Visual Genome dataset does not contain mask annotation, it means this evaluation type is not available for final model, but you can visualize predictions.

## Feature Extraction

Bellow represented base command for feature extraction:
```bash
python tools/extact_features.py <model_config> <model_checkpoint> --out_tsv result.tsv --data_dir <image_dir>
```
Optionally you can provide `--max_obj` - maximum number of predicted objects for image (default 100) and specify `--device` device id for feature extraction if multiple gpu represented in the system.
Features will be stored in tsv format. 
Generated tsv file fields:
* `img_id` - image identification  in the dataset (usually file name)
* `img_h` - image height
* `img_w` - image width
* `objects_id` - detection labels for objects
* `objects_conf` - confidence of predicted objects
* `num_boxes` - number predicted objects
* `features` - Mask RCNN or Faster RCNN features (depends on used model)
* `boxes` - normalized detection boxes coordinates (in range [0, 1]) in format (x_min, y_min, x_max, y_max).
* `masks` - predicted masks (if Mask RCNN used)
* `bbox_features` - detection bounding box ROI features (if Mask RCNN used)

