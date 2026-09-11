# PointNet / PointNet++ / PointNeXt in PyTorch

This repository provides an implementation for **PointNet**, **PointNet++**, and extended **PointNeXt** architecture variants in PyTorch.

The framework is highly optimized for both standard academic benchmarks and complex industrial 3D point cloud tasks. It specifically tackles the challenges of industrial datasets, which are often characterized by small sample sizes, severe class imbalances, and high geometric noise.

## Project Background and Academic Context

This tool is part of the experimental methodology of the following master's thesis:

- **Thesis Title**: A Study on the Classification and Process Prediction of Gating Systems for Investment Casting of A356 Aluminum Alloy
- **Author**: HSU, WEN-HO
- **Advisor**: CHEN, TZUNG-MING
- **Institution**: National Changhua University of Education
- **Department**: Department of Electrical and Mechanical Technology
- **Degree**: Master's Thesis
- **Oral Defense Date**: 2026-07-10
- **Permanent URL**: [https://hdl.handle.net/11296/32a644](https://hdl.handle.net/11296/32a644)

**Keywords**: A356 aluminum alloy, investment casting, automated machine learning (AutoML), 3D point cloud semantic segmentation.

The research aims to establish a gating system classification model and process prediction system for A356 aluminum alloy investment casting. This repository provides the deep learning backbone for **runner system identification and gate localization** on industrial casting point clouds. The modifications described below enable the network to handle multi-modal inputs (coordinates, normals, and continuous soft labels) and to address the severe class imbalance inherent in gate regions, directly supporting the thesis's segmentation and regression tasks.

### Supported Tasks:
*   **Global Object Classification**
*   **Part Segmentation**
*   **Semantic Segmentation**
*   **Binary Segmentation** (Hard Labels)
*   **Heatmap Regression** (Soft Labels)

---

## PointNeXt Extension & Structural Modifications

This repository contains significant modifications tailored for **runner system identification and gate localization in precision investment casting**. To successfully bridge raw CAD data with deep learning, the following structural and dimensional enhancements were implemented:

*   **Dynamic Tensor Dimension Alignment:** Adapted the core architecture to process varying tensor dimensions (e.g., N×3, N×6, or N×7 formats). This ensures that multi-modal inputs—such as XYZ coordinates, surface normals, and continuous soft labels—are perfectly aligned into standardized spatial features that facilitate deep learning convergence[cite: 1].
*   **Automated Soft-Labeling Mechanism:** Replaced traditional binary masking with a continuous soft-labeling mechanism, supplemented by 3D geometric heatmaps, allowing the network to learn the smooth gradient transitions of gate contact areas[cite: 1].
*   **Customized Prediction Heads:** Modified the segmentation and regression output layers to handle arbitrary dimensional scaling, avoiding tensor mismatch errors during multi-scale feature propagation (FP) and set abstraction (SA).

---

## Key Features

### 1. Dynamic Architecture Generation
Includes a permutation generator script that can automatically build **36 neural network architecture variants** with a single click. These variants span multiple design dimensions, facilitating comprehensive ablation studies:
*   **Multi-Scale Grouping/Aggregation**: `MSG` / `SSG` / `MRG`
*   **Feature Refinement Modules**: `BottleneckMLP` / `InvertedResidual` / `Increasing`
*   **Spatial Transformer Networks**: `STN3d` / `NoSTN`
*   **Task Types**: `Seg` / `Regress`

### 2. Hybrid Loss Functions
*   **Segmentation Task:** Integrates `Dice Loss` and `Focal Loss` to effectively address the severe class imbalance caused by the extremely small proportion of gate point clouds.
*   **Regression Task:** Utilizes `Region-Normalized MSE` combined with `Soft Dice Loss`. This forces the optimization weights of the foreground (gates) and background (main body) to be balanced, guiding the model to accurately predict continuous thermal decay values.

### 3. Smart Rewind & Punish Protocol
A built-in automated overfitting protection mechanism. When validation metrics (e.g., MAE or mIoU) stagnate for a consecutive number of epochs, the system automatically triggers:
*   **Rewind**: Loads the historical best model weights.
*   **Punish**: Increases regularization (doubles the Weight Decay).
*   **Warm Restart**: Restarts and halves the learning rate (via Cosine Annealing) to force the model out of local optima.

### 4. Multi-dimensional Evaluation & Visualization
*   **Spatial Metrics**: In addition to traditional mIoU, it calculates the physical Euclidean distance error between the predicted centroid and the ground truth centroid (`Shift Error` / `Weighted Shift Error`).
*   **Automated Rendering**: Automatically exports 3D multi-view comparison plots (Prediction vs Ground Truth), decision threshold scanning curves (Threshold-mIoU Curve), hardest class rankings, and normalized confusion matrix heatmaps during training.

---

## Installation

The latest codes are tested on Ubuntu 16.04, CUDA 10.1, PyTorch 1.6, and Python 3.7.

```bash
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
Note: The PointNeXt extension shares the same environment and has no additional mandatory dependencies beyond the original implementation.

Directory Structure & Core Scripts
train_classification_keypoint.py: Executes the global feature classification task for runner systems.

train_partseg_nocls_segmentation_smart_v5.py: Executes the binary semantic segmentation task (generates hard masks).

train_partseg_nocls_regress_v3.py: Executes the thermal field regression task (predicts continuous contact intensity).

Model Generation Script: Dynamically writes architecture files (e.g., exp_pointnext_part_seg_nocls_msg_invertedresidualmlp_nostn_regress.py) into the models/ directory.

data_utils/: Data preprocessing utilities.

visualizer/: C++ and Python visualization tools.

log/: Training logs, checkpoints, and output reports.

Quick Start Guide
1. Generate the Model Architecture Library
Before starting the training, run the generation script to build all necessary network architectures:

bash
python generate_models.py
Once executed, 36 architecture .py files will be automatically generated under the models/ folder.

2. Classification (ModelNet10/40)
Data Preparation: Download alignment ModelNet and save in data/modelnet40_normal_resampled/. If you want to use offline processing of data to accelerate training, use --process_data in the first run. For ModelNet10, use --num_category 10.

bash
# Example: pointnet2_ssg without normal features
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
python test_classification.py --log_dir pointnet2_cls_ssg

# Example: pointnet2_ssg with normal features
python train_classification.py --model pointnet2_cls_ssg --use_normals --log_dir pointnet2_cls_ssg_normal

# Example: pointnet2_ssg with uniform sampling
python train_classification.py --model pointnet2_cls_ssg --use_uniform_sample --log_dir pointnet2_cls_ssg_fps
Model	Accuracy
PointNet (Official)	89.2
PointNet2 (Official)	91.9
PointNet (Pytorch without normal)	90.6
PointNet (Pytorch with normal)	91.4
PointNet2_SSG (Pytorch without normal)	92.2
PointNet2_SSG (Pytorch with normal)	92.4
PointNet2_MSG (Pytorch with normal)	92.8
3. Part Segmentation (ShapeNet)
Data Preparation: Download alignment ShapeNet and save in data/shapenetcore_partanno_segmentation_benchmark_v0_normal/.

bash
# Example: pointnet2_msg
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
python test_partseg.py --normal --log_dir pointnet2_part_seg_msg
Model	Instance avg IoU	Class avg IoU
PointNet (Official)	83.7	80.4
PointNet2 (Official)	85.1	81.9
PointNet (Pytorch)	84.3	81.1
PointNet2_SSG (Pytorch)	84.9	81.8
PointNet2_MSG (Pytorch)	85.4	82.5
4. Semantic Segmentation (S3DIS)
Data Preparation: Download the 3D indoor parsing dataset (S3DIS) and save in data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/.

bash
cd data_utils
python collect_indoor3d_data.py
Run:

bash
# Example: pointnet2_ssg
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5 --visual
Visualization results will be saved in log/sem_seg/pointnet2_sem_seg/visual/.

Model	Overall Acc	Class avg IoU	Checkpoint
PointNet (Pytorch)	78.9	43.7	40.7MB
PointNet2_ssg (Pytorch)	83.0	53.5	11.2MB
5. Industrial Casting Tasks (PointNeXt)
Specify the generated network architecture using the --model parameter and provide the corresponding dataset path.

Classification Task

bash
python train_classification_keypoint.py --model pointnet_cls --data_dir data/cast_dataset --epoch 200
Segmentation / Regression Task

Supports --use_smart_rewind to activate the Smart Rewind mechanism.

bash
python train_partseg_nocls_regress_v3.py \
    --model exp_pointnext_part_seg_nocls_msg_invertedresidualmlp_nostn_regress \
    --data_dir data/cast_dataset_gate \
    --epoch 500 \
    --batch_size 16 \
    --optimizer AdamW \
    --use_smart_rewind \
    --patience 15
For binary segmentation, use train_partseg_nocls_segmentation_smart_v5.py with similar arguments.

Outputs & Report Interpretation:

Upon completion of training, all results are saved in timestamped folders under the log/part_seg/ or log/classification/ directories. These include:

checkpoints/: Stores the best_model.pth.

out_data/: Contains the .txt prediction results (prediction/) and labels (ground_truth/), restored to absolute coordinates and original filenames.

visual_results/: Holds 3D heatmap renderings and evaluation metric charts.

*.csv: Detailed inference reports (e.g., inference_full_report.csv, class_error_summary.csv) documenting single-sample performance, hit status, and error rates per class.

Visualization Tools
Using show3d_balls.py

bash
# build C++ code for visualization
cd visualizer
bash build.sh

# run one example
python show3d_balls.py
References & Credits
halimacc/pointnet3

fxia22/pointnet.pytorch

charlesq34/PointNet

charlesq34/PointNet++

Citation
If you find this repo useful in your research, please consider citing the original works and our adaptations.

Original Works
bibtex
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}

@InProceedings{yan2020pointasnl,
  title={PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling},
  author={Yan, Xu and Zheng, Chaoda and Li, Zhen and Wang, Sheng and Cui, Shuguang},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
Thesis Citation
If this tool is helpful to your research, please also cite:

HSU, W.-H. (2026). A Study on the Classification and Process Prediction of Gating Systems for Investment Casting of A356 Aluminum Alloy (Master's thesis). National Changhua University of Education, Department of Electrical and Mechanical Technology, Changhua City. Retrieved from https://hdl.handle.net/11296/32a644

text

You can replace your existing `README.md` with the above content. The thesis background is integrated after the introduction, and the citation is added at the end alongside the original references.
