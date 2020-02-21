Implementation Clothflow (Here is a paper's [link](http://openaccess.thecvf.com/content_ICCV_2019/papers/Han_ClothFlow_A_Flow-Based_Model_for_Clothed_Person_Generation_ICCV_2019_paper.pdf).)
# Stage 0
We used a MVC dataset. Please download this first.
* ## Resize
Image's original size is (1920, 2240), but this is too big to learn for computer. So, we need to resize this.

Before run this code, you must be set `TARGET_SIZE` and `base_dir`.

```python
python resize.py
```

* ## Segmentation
Use opensource for segmentation. You must save images in 'p','1','2',3',... in image folder. 

We use LIP_JPPNet.

* ## Pose
Use opensource for pose. You must save images in 'p','1','2',3',... in image folder. 

We use pytorch_Realtime_Multi-Person_Pose_Estimation.

* ## Crop
Crop clothes from person's image.

Before run this code, you must be set `TARGET_SIZE`, `IS_TOPS`, `base_dir`. The meaning of `IS_TOPS` is whether you are learning tops or bottoms.
```python
python crop.py
```
* ## Split
Split train data and test data.

Before run this code, you must be set `base_dir`.

```python
python stage0/split.py
```

* ## Make pair txt
Make pair in same clothes image.

Before run this code, you must be set `base_dir`.
```python
python mkpairtxt.py
```

# Stage 1
**Change condition cloth mask to target cloth mask**
* ## Train
```python
python stage1/train.py
```
* ## Warped_Mask
```python
python stage1/test.py
```
# Stage 2
**Warp**
* ## Train
```python
python stage2/train.py
```
* ## Warped_Cloth
```python
python stage2/test.py
```
# Stage 3
**wear cloth**
[]
* ## Train
```python
python stage3/train.py
```
* ## Final Result
```python 
python stage3/test.py
```

# Inference
```python
python inference.py
```