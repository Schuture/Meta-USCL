# Meta-USCL
This is the repository for Meta **U**ltra**S**ound **C**ontrastive **L**earning (Meta-USCL), implemented with PyTorch. 

### 1. Brief Introduction to this Method

SimCLR + **P**ositive **P**air **I**nterpolator (PPI, to generate positive pairs from an ultrasound video clip) + **C**ontrastive **M**eta **W**eight Net (CMW-Net, a meta-learning-driven positive pair weighting network).

Meta-USCL ([TMI paper](https://ieeexplore.ieee.org/abstract/document/9980429)) is an improved version of USCL ([MICCAI paper](https://github.com/983632847/USCL)). The key difference is the new CMW-Net, which is a generic method for weighting the random generated positive pairs (they tend to have highly diverse benefit for learning semantic consistency).


### 2. Framework of Meta-USCL

![Framework](https://github.com/Schuture/Meta-USCL/blob/main/metaUSCL.png)


### 3. The Learned Weighting Scheme

(1) For different subsets (domains) of our pre-training dataset US-4, CMW-Net gives different weight distributions.

<img src="https://github.com/Schuture/Meta-USCL/blob/main/histogram_weight.png" width = "600" height = "450" alt="Weight distribution across different domains" align=center />

(2) The illustration of weights and similarities between an initial frame and other frames. Red boxes highlight the changing semantic marks (e.g., hepatic vein, portal vein and inferior vena cava) from the initial frame, green boxes denote the textural dissimilarities (gas in the lungs and a small portal vein section) in the following frames.

![The relationship between similarity and weight](https://github.com/Schuture/Meta-USCL/blob/main/Sim_vs_weight.png)


### 4. Quick Start

#### 4.1 Fine-tuning directly with the pre-trained model
1. Download the 5 fold cross validation [POCUS](https://drive.google.com/file/d/111lHpStoY_gYMhCQ-Yt95AreDx0G7-2R/view?usp=sharing) dataset
2. Run the demo with
```
# Using 64x64 pre-trained model to fine-tune on 64x64 image size
python eval_pretrained_model_on_POCUS/train.py --path model_ckpt/resnet18_64.pth --input_shape 64

# Using 224x224 pre-trained model to fine-tune on 224x224 image size
python eval_pretrained_model_on_POCUS/train.py --path model_ckpt/resnet18_224.pth --input_shape 224
```

Name | Pre-trained size | Epochs | Project head | Classifier | Accuracy | Download
---  |:---------:|:---------:|:---------:|:---------:|:---------:|:---------:
ShuffleNet v2 | 64x64 | 300 | Yes | Yes | 87.3% |  
ShuffleNet v2 | 224x224 | 300 | Yes | Yes | 90.3% |  
ResNet-18 | 64x64 | 300 | Yes | Yes | 90.9% |  [link](https://drive.google.com/file/d/1yKimuJUwkp3qRirsnIhJtbBv47wPouf7/view?usp=sharing)
ResNet-18 | 224x224 | 300 | Yes | Yes | 94.6% |  [link](https://drive.google.com/file/d/1yKimuJUwkp3qRirsnIhJtbBv47wPouf7/view?usp=sharing)


#### 4.2 Train Your Own Model
1. Download the Butterfly ([Baidu pan](https://pan.baidu.com/s/1tQtDzoditkTft3LMeDfGqw) Pwd:butt, [Google drive](https://drive.google.com/file/d/1zefZInevopumI-VdX6r7Bj-6pj_WILrr/view?usp=sharing)) dataset 
2. Train the USCL model with
```
python train.py
```

### 5. Apply Pre-trained Models with Mask-RCNN for Tumor Segmentation
1. Download pre-trained [ResNet-18](https://drive.google.com/file/d/1yKimuJUwkp3qRirsnIhJtbBv47wPouf7/view?usp=sharing) (same as above).
2. Unzip the UDIAT-B breast tumor segmentation dataset at *eval_mask_rcnn_on_UDIATB/UDIAT_Dataset_B.zip*.
3. Fine-tune pre-trained model with
```
python eval_mask_rcnn_on_UDIATB/run.py --dataset_dir eval_mask_rcnn_on_UDIATB/UDIAT_Dataset_B --path model_ckpt/resnet18_224.pth
```



#### 6. Environment
The code is developed with an Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz and a single Nvidia Tesla V100 GPU.

The install script *requirements.txt* has been tested on an Ubuntu 18.04 system.

:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:


### 7. License

Licensed under an MIT license.


### 8. Citation

This work is published at IEEE Transaction on Medical Imaging. If you find the code and dataset useful in your research, please consider citing:

    @article{chen2022generating,
        title={Generating and Weighting Semantically Consistent Sample Pairs for Ultrasound Contrastive Learning},
        author={Chen, Yixiong and Zhang, Chunhui and Ding, Chris HQ and Liu, Li},
        journal={IEEE Transactions on Medical Imaging},
        year={2022},
        publisher={IEEE}
    }

    @inproceedings{Chen2021MICCAI,
        title={USCL: Pretraining Deep Ultrasound Image Diagnosis Model through Video Contrastive Representation Learning},
        author = {Yixiong Chen, and Chunhui Zhang, and Li Liu, and Cheng Feng, and Changfeng Dong, and Yongfang Luo, and Xiang Wan},
        journal = {MICCAI},
        year = {2021}
      }


    @article{born2021accelerating,
        title={Accelerating detection of lung pathologies with explainable ultrasound image analysis},
        author={Born, Jannis and Wiedemann, Nina and Cossio, Manuel and Buhre, Charlotte and Br{\"a}ndle, Gabriel and Leidermann, Konstantin and Aujayeb, Avinash and Moor, Michael and Rieck, Bastian and Borgwardt, Karsten},
        journal={Applied Sciences},
        pages={672},
        year={2021},
    }


### 9. Contact
Feedbacks and comments are welcome! Feel free to contact us via [yixiongchen@link.cuhk.edu.cn](mailto:yixiongchen@link.cuhk.edu.cn) or [andyzhangchunhui@gmail.com](mailto:andyzhangchunhui@gmail.com) or [liuli@cuhk.edu.cn](mailto:liuli@cuhk.edu.cn).

Enjoy!









