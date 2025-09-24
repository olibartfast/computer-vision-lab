# Computer Vision Lab

This repository, which is still a work in progress, serves as a reference guide and outline covering the fundamentals and techniques of computer vision. Topics range from basic image processing and feature detection to cutting-edge neural rendering and SLAM systems.

## 📁 Repository Structure

```
computer-vision-lab/
├─ cvlab/                # Core library modules
│  ├─ image/            # Image processing fundamentals
│  ├─ features/         # Feature detection and matching
│  ├─ geometry/         # Geometric computer vision
│  ├─ tracking/         # Motion analysis and tracking
│  ├─ slam/             # SLAM implementations
│  ├─ deep/             # Deep learning approaches
│  └─ deploy/           # Model optimization and deployment
├─ examples/             # Runnable scripts and notebooks
│  ├─ basic/            # Fundamental CV examples
│  ├─ deep_learning/    # Neural network examples
│  └─ applications/     # Real-world applications
├─ data/                 # Sample datasets and inputs
├─ tests/               # Unit and integration tests
├─ docs/                # Documentation and tutorials
├─ requirements.txt     # CPU dependencies
├─ requirements-gpu.txt # GPU dependencies
├─ environment.yml      # Conda environment
└─ README.md
```

## 📋 Table of Contents

- [Image Processing Fundamentals](#image-processing-fundamentals)
- [Feature Detection and Matching](#feature-detection-and-matching)
- [Geometric Computer Vision](#geometric-computer-vision)
- [3D Computer Vision](#3d-computer-vision)
- [Motion Analysis and Tracking](#motion-analysis-and-tracking)
- [SLAM (Simultaneous Localization and Mapping)](#slam-simultaneous-localization-and-mapping)
- [Deep Learning for Computer Vision](#deep-learning-for-computer-vision)
- [Model Optimization and Deployment](#model-optimization-and-deployment)
- [Applications](#applications)

## Image Processing Fundamentals

Basic image operations and transformations:
- **Image Function Operations**: Pixel-level operations, intensity transformations
- **Geometric/Affine Transforms**: Scaling, rotation, shearing, translation, warping
- **Homogeneous Coordinates**: Projective geometry representations
- **Histogram Processing**: Histogram equalization, CLAHE, histogram matching
- **Intensity Transformations**: Gamma correction, log transforms, contrast stretching
- **Filtering and Enhancement**: Noise reduction, sharpening, edge detection
- **Morphological Operations**: Erosion, dilation, opening, closing
- **Color Space Transformations**: RGB, HSV, LAB, grayscale conversions
- **Image Restoration**: Deblurring, inpainting, super-resolution

## Feature Detection and Matching

Image comparison and feature matching techniques:
- **Corner Detection**: Harris corners, FAST, Shi-Tomasi
- **Keypoint Descriptors**: SIFT, SURF, ORB, BRIEF
- **Feature Matching**: Brute-force, FLANN-based matching
- **Statistical Measures**: Image moments, similarity metrics, distance functions
- **Distribution-Based Methods**: Bhattacharyya distance, KL/JS divergence, Chi-square, Earth Mover's Distance
- **Cross Correlation**: Template matching, normalized cross-correlation
- **Convolution vs Correlation**: 2D convolution, separable filters (note: OpenCV filter2D implements correlation)

## Geometric Computer Vision

Mathematical foundations of computer vision:
- **Camera Models**: Pinhole camera, lens distortion models
- **Camera Calibration**: Intrinsic and extrinsic parameter estimation
- **RANSAC**: Robust model fitting, outlier rejection, geometric estimation
- **Homography**: Plane-to-plane transformations, image rectification
- **Fundamental Matrix**: Epipolar geometry between two views
- **Essential Matrix**: Calibrated camera epipolar geometry
- **Triangulation**: 3D point reconstruction from multiple views
- **Pose Estimation**: PnP algorithms, camera localization
- **Robust Estimation**: M-estimators, MSAC, PROSAC, LO-RANSAC

## 3D Computer Vision

Three-dimensional scene understanding:
- **Stereo Vision**: Disparity estimation, depth maps
- **Structure from Motion (SfM)**: COLMAP, 3D reconstruction from image sequences
- **Multi-view Stereo**: Dense 3D reconstruction, PatchMatch (classical and deep)
- **TSDF Fusion**: KinectFusion, volumetric integration
- **Neural Radiance Fields (NeRF)**: Neural volume rendering, novel view synthesis
- **Gaussian Splatting**: 3D Gaussian primitives, real-time differentiable rendering
- **Point Cloud Processing**: Registration, filtering, segmentation
- **3D Object Recognition**: Pose estimation, model fitting
- **Volume Rendering**: Ray casting, marching cubes, neural rendering
- **Novel View Synthesis**: Image-based rendering, view interpolation
- **Coordinate Systems**: World, camera, and image coordinate transformations

## Motion Analysis and Tracking

Temporal analysis and object tracking:
- **Background Subtraction**: GMM, running average, ViBe, deep learning approaches
- **Optical Flow**: Lucas-Kanade, Horn-Schunck, RAFT, GMFlow methods
- **Visual Odometry**: Camera motion estimation from image sequences
- **Object Tracking**: KCF/CSRT, SiamFC/RPN, SORT/DeepSORT, ByteTrack, BoT-SORT
- **Motion Detection**: Frame differencing, temporal gradients
- **Kalman Filtering**: State estimation and prediction
- **Motion Segmentation**: Separating moving objects from background
- **Action Recognition**: SlowFast, VideoMAE, TimeSformer, V-JEPA, temporal pattern analysis
- **Multi-Object Tracking**: Data association, trajectory management, re-identification integration

## SLAM (Simultaneous Localization and Mapping)

Real-time localization and mapping systems:
- **Visual SLAM**: MonoSLAM, PTAM, ORB-SLAM, ORB-SLAM3
- **Direct Methods**: DSO, LSD-SLAM, photometric approaches
- **Visual-Inertial SLAM**: VINS-Mono/Fusion, OKVIS, sensor fusion approaches
- **Neural SLAM**: NICE-SLAM, iMAP, DROID-SLAM, learning-based approaches
- **Loop Closure Detection**: Place recognition algorithms, bag-of-words
- **Bundle Adjustment**: Optimization of camera poses and 3D points
- **Keyframe Management**: Efficient map representation
- **Dense SLAM**: Real-time dense reconstruction, ElasticFusion
- **Semantic SLAM**: Object-aware mapping

## Deep Learning for Computer Vision

Neural network approaches:
- **Convolutional Neural Networks**: CNN architectures for vision tasks
- **Object Detection**: YOLO (v5/v8/Next), R-CNN, DETR/Deformable DETR, RT-DETR
- **Semantic Segmentation**: FCN, U-Net, DeepLab, SegFormer
- **Instance Segmentation**: Mask R-CNN, YOLACT, Mask2Former
- **Panoptic Segmentation**: Unified instance and semantic segmentation
- **Foundation Models**: SAM (Segment Anything), CLIP, DINOv2
- **Depth Estimation**: MonoDepth, stereo matching networks
- **Neural Rendering**: NeRF variants (Instant-NGP, Mip-NeRF), neural volume rendering
- **3D Deep Learning**: PointNet/PointNet++, KPConv, MinkowskiEngine, Point Transformers
- **Video Understanding**: 3D CNNs (I3D, C3D), two-stream networks, video transformers
- **Visual Transformers**: ViT, Swin Transformer, DeiT, DINO applications
- **Self-Supervised Learning**: Contrastive learning, masked modeling, representation learning
- **Generative Models**: Stable Diffusion, ControlNet, video diffusion models
- **Neural SLAM**: Learning-based SLAM approaches

## Model Optimization and Deployment

Techniques for efficient deployment of computer vision models:
- **Post-Training Quantization**: Model compression (2x-4x) without retraining, INT8/FP16 optimization
- **Quantization-Aware Training**: Training with quantization simulation for improved accuracy
- **Pruning**: Structured and unstructured weight removal, magnitude-based pruning
- **Knowledge Distillation**: Teacher-student training, model compression, soft targets
- **Sparsity**: Sparse tensor storage, structured sparsity, magnitude pruning
- **Dynamic Inference**: Early-exiting, token pruning (for ViTs), multi-scale and dynamic resolution
- **Graph Optimization**: Operator fusion, kernel autotuning, TVM compilation
- **Hardware Optimization**: TensorRT, ONNX Runtime, mobile frameworks (Core ML, TensorFlow Lite)
- **Model Serving**: Triton Inference Server, TorchServe, FastAPI/gRPC, batch optimization

## Applications

Real-world computer vision applications:
- **Autonomous Driving**: Perception systems, lane detection
- **Robotics**: Navigation, manipulation, human-robot interaction
- **Augmented Reality**: Marker tracking, pose estimation
- **Medical Imaging**: Image analysis, diagnostic tools
- **Industrial Inspection**: Quality control, defect detection
- **Surveillance**: Activity recognition, anomaly detection

## 📊 Evaluation Metrics

Common metrics for computer vision tasks:
- **Image Quality**: PSNR, SSIM, LPIPS, FID for image generation/restoration
- **Object Detection**: mAP, mAP@50, mAP@75, precision, recall, F1-score
- **Segmentation**: mIoU, pixel accuracy, dice coefficient, boundary F1-score
- **Depth Estimation**: AbsRel, RMSE, δ1/δ2/δ3 accuracy thresholds
- **Optical Flow**: End-Point Error (EPE), angular error, AUC
- **Tracking**: IDF1, HOTA, MOTA, MOTP, identity switches
- **Action Recognition**: Top-1/Top-5 accuracy, temporal localization metrics
- **3D Reconstruction**: Chamfer distance, F-score, completeness, accuracy

## 🛠️ Technologies Used

- **Programming Languages**: Python, C++
- **Libraries**: OpenCV, NumPy, SciPy, scikit-image, PIL/Pillow
- **Deep Learning**: PyTorch, TensorFlow, Keras, Hugging Face Transformers
- **3D Processing**: Open3D, PCL (Point Cloud Library), trimesh
- **Optimization**: TensorRT, ONNX, OpenVINO, TensorFlow Lite
- **Visualization**: Matplotlib, Plotly, VTK, Weights & Biases
- **Experiment Tracking**: TensorBoard, Weights & Biases, MLflow



## 📚 Resources

- [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/) by Richard Szeliski
- [Multiple View Geometry](https://www.robots.ox.ac.uk/~vgg/hzbook/) by Hartley & Zisserman
- [OpenCV Documentation](https://docs.opencv.org/)
- [SLAM Literature](https://github.com/kanster/awesome-slam)

