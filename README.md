#Merging Neural Radiance Fields for Clothes Fitting

This is a project researching virtual clothes fitting using Neural Radiance Fields (NeRF). This project leverages state-of-the-art computer vision techniques to create 3D models of both humans and clothing items, merging them for realistic virtual try-on applications.

## Overview

The project combines several cutting-edge technologies:

1. **Neural Radiance Fields (NeRF)** for high-quality 3D reconstruction from 2D images
2. **Segment Anything Model (SAM)** for precise clothing segmentation from images
3. **COLMAP** for structure-from-motion and camera parameter extraction
4. **Nerf2Mesh** for generating 3D meshes from neural representations

The pipeline enables users to visualize how clothing items would look on a person without the need for physical fitting, bringing significant advancements to e-commerce, fashion design, and virtual reality applications.

## Features

- **Monocular Video Processing**: Generate 3D models from standard video input
- **High-Quality Segmentation**: Precisely separate clothing items from human models
- **Physics-Based Cloth Simulation**: Realistic cloth behavior and fitting
- **Mesh Generation**: Convert neural representations to standard 3D meshes compatible with popular 3D software
- **Background Removal**: Clean isolation of subjects from complex backgrounds
- **CUDA Acceleration**: Optimized performance with GPU acceleration

## Pipeline

Our pipeline consists of the following stages:

1. **Data Acquisition**: Capture monocular videos of humans and clothing items
2. **Frame Extraction**: Generate frames from videos using COLMAP
3. **Camera Parameter Estimation**: Extract intrinsic and extrinsic camera parameters
4. **Segmentation**: Apply SAM to isolate clothing from each frame
5. **NeRF Training**: Generate neural radiance field representations
6. **Mesh Generation**: Convert NeRF models to 3D meshes
7. **Simulation & Fitting**: Apply physics-based simulation for realistic cloth draping
8. **Rendering**: Create final visualizations of the fitted clothing

## Installation

```bash
# Clone the repository
git clone https://github.com/aaryansarnaik7832/neuralRadianceFieldsForClothesFitting.git
cd neuralRadianceFieldsForClothesFitting

# Create a conda environment
conda create -n nerf-clothes python=3.8
conda activate nerf-clothes

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (with CUDA support)
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install COLMAP (follow instructions for your OS)
# For Ubuntu:
apt-get install colmap
```

## Usage

### Data Preparation

```bash
# Generate frames from video and prepare COLMAP data
python colmap2nerf.py --video path/to/your/video.mp4 --run_colmap --video_fps 30 --colmap_matcher sequential
```

### Clothing Segmentation

```bash
# Segment clothing items from frames
python clothes-seg.py --input_dir path/to/frames --output_dir path/to/segmented --resize_images --image_size 512_512
```

### Background Removal

```bash
# Remove backgrounds from images
python remove_bg.py path/to/images
```

### NeRF Training and Mesh Generation

```bash
# Train NeRF and generate mesh
python nerf2mesh.py path/to/data --workspace path/to/output --stage 0 --O
```

### Downscale Images (if needed)

```bash
# Downscale images for faster processing
python downscale.py path/to/data --downscale 4
```

## Datasets

We utilize the following datasets:

- **HumanNeRF Dataset**: Monocular videos of moving individuals, allowing for free-viewpoint rendering of human subjects
- **Deep Fashion Dataset**: A large-scale clothes database with diverse clothing items and styles

## Results

Our approach successfully generates high-quality 3D models of clothing items that can be realistically fitted to human models. The system handles various clothing types and body shapes while maintaining physical accuracy.

## Technical Details

### Neural Radiance Fields (NeRF)

We employ NVIDIA's Instant NeRF (NGP) for efficient 3D reconstruction, significantly reducing training time compared to traditional NeRF implementations.

### Segment Anything Model (SAM)

SAM provides state-of-the-art segmentation capabilities, allowing us to precisely isolate clothing items from complex backgrounds and human subjects.

### Physics-Based Simulation

Our approach incorporates physical properties of fabrics, ensuring that the virtual try-on results accurately represent how clothing would drape and fit in reality.

## Acknowledgments

- [NVIDIA Instant NGP](https://github.com/NVlabs/instant-ngp) for their efficient NeRF implementation
- [Segment Anything Model](https://github.com/facebookresearch/segment-anything) for state-of-the-art segmentation
- [COLMAP](https://colmap.github.io/) for structure-from-motion and multi-view stereo
- [Nerf2Mesh](https://github.com/ashawkey/nerf2mesh) for NeRF to mesh conversion
- Deep Fashion dataset and HumanNeRF dataset for providing valuable training data

## Team

- **Aaryan Mahesh Sarnaik**
- **Praney Goyal**
- **Travis Hughes**

Project done under the supervision of Dr. Robert Collins of the Pennsylvania State University.
