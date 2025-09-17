# Computer Vision Lab

A comprehensive collection of computer vision algorithms, techniques, and implementations covering fundamental to advanced topics in the field.

## 📋 Table of Contents

- [Image Processing Fundamentals](#image-processing-fundamentals)
- [Feature Detection and Matching](#feature-detection-and-matching)
- [Geometric Computer Vision](#geometric-computer-vision)
- [3D Computer Vision](#3d-computer-vision)
- [Motion Analysis and Tracking](#motion-analysis-and-tracking)
- [SLAM (Simultaneous Localization and Mapping)](#slam-simultaneous-localization-and-mapping)
- [Deep Learning for Computer Vision](#deep-learning-for-computer-vision)
- [Applications](#applications)

## Image Processing Fundamentals

Basic image operations and transformations:
- **Image Function Operations**: Pixel-level operations, intensity transformations
- **Linear Transforms**: Scaling, rotation, shearing, translation
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
- **Distribution-Based Methods**: Cross entropy, histogram comparison
- **Cross Correlation**: Template matching, normalized cross-correlation
- **Convolution**: 2D convolution, separable filters

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
- **Geometric Transformations**: Affine, projective, similarity transforms

## 3D Computer Vision

Three-dimensional scene understanding:
- **Stereo Vision**: Disparity estimation, depth maps
- **Structure from Motion (SfM)**: 3D reconstruction from image sequences
- **Multi-view Stereo**: Dense 3D reconstruction
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
- **Optical Flow**: Lucas-Kanade, Horn-Schunck methods
- **Visual Odometry**: Camera motion estimation from image sequences
- **Object Tracking**: Mean-shift, CamShift, particle filters, Kalman tracking
- **Motion Detection**: Frame differencing, temporal gradients
- **Kalman Filtering**: State estimation and prediction
- **Motion Segmentation**: Separating moving objects from background
- **Action Recognition**: Temporal pattern analysis
- **Multi-Object Tracking**: Data association, trajectory management

## SLAM (Simultaneous Localization and Mapping)

Real-time localization and mapping systems:
- **Visual SLAM**: MonoSLAM, PTAM, ORB-SLAM
- **Visual-Inertial SLAM**: Sensor fusion approaches
- **Loop Closure Detection**: Place recognition algorithms
- **Bundle Adjustment**: Optimization of camera poses and 3D points
- **Keyframe Management**: Efficient map representation
- **Dense SLAM**: Real-time dense reconstruction
- **Semantic SLAM**: Object-aware mapping

## Deep Learning for Computer Vision

Neural network approaches:
- **Convolutional Neural Networks**: CNN architectures for vision tasks
- **Object Detection**: YOLO, R-CNN, SSD implementations
- **Semantic Segmentation**: FCN, U-Net, DeepLab
- **Instance Segmentation**: Mask R-CNN, YOLACT
- **Depth Estimation**: MonoDepth, stereo matching networks
- **Neural Rendering**: NeRF variants (Instant-NGP, Mip-NeRF), neural volume rendering
- **3D Deep Learning**: 3D CNNs, point cloud networks, voxel-based approaches
- **Visual Transformers**: Vision Transformer (ViT) applications
- **Generative Models**: GANs for image synthesis, diffusion models
- **Neural SLAM**: Learning-based SLAM approaches

## Applications

Real-world computer vision applications:
- **Autonomous Driving**: Perception systems, lane detection
- **Robotics**: Navigation, manipulation, human-robot interaction
- **Augmented Reality**: Marker tracking, pose estimation
- **Medical Imaging**: Image analysis, diagnostic tools
- **Industrial Inspection**: Quality control, defect detection
- **Surveillance**: Activity recognition, anomaly detection

## 🛠️ Technologies Used

- **Programming Languages**: Python, C++
- **Libraries**: OpenCV, NumPy, SciPy, scikit-image
- **Deep Learning**: TensorFlow, PyTorch, Keras
- **3D Processing**: Open3D, PCL (Point Cloud Library)
- **Visualization**: Matplotlib, Plotly, VTK

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/olibartfast/computer-vision-lab.git
cd computer-vision-lab

# Install dependencies
pip install -r requirements.txt

# Run examples
python examples/basic_image_processing.py
```

## 📚 Resources

- [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/) by Richard Szeliski
- [Multiple View Geometry](https://www.robots.ox.ac.uk/~vgg/hzbook/) by Hartley & Zisserman
- [OpenCV Documentation](https://docs.opencv.org/)
- [SLAM Literature](https://github.com/kanster/awesome-slam)

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
