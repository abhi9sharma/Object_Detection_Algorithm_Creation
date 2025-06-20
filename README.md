# Object Detection with U-Net and YOLO-inspired Architecture

This repository contains the implementation of a novel object detection model that combines a U-Net-based feature extractor with a YOLO-inspired classification head, as described in the paper "Advancements in Object Detection Algorithms: Evaluating, Enhancing and Envisioning" by Abhinav Sharma.

## Project Overview

The model is designed to address challenges in object detection, such as detecting small objects, balancing speed and accuracy, and reducing computational requirements. It leverages the U-Net architecture for feature extraction and a custom classification head inspired by YOLO for predicting bounding boxes and class probabilities.

### Features
- **U-Net Backbone**: Extracts multi-scale features using a downsampling and upsampling path with skip connections.
- **YOLO-inspired Classification**: Divides the image into a 7x7 grid to predict class probabilities, object confidence, and bounding box coordinates.
- **Custom Dataset**: Supports a dataset with images and bounding box annotations in a CSV file.
- **PyTorch Implementation**: Built using PyTorch for flexibility and GPU acceleration.

## Requirements

To run the code, install the following dependencies:

```bash
pip install torch torchvision pandas numpy pillow opencv-python matplotlib
```

Ensure you have a CUDA-enabled GPU for faster training, though the code will fall back to CPU if no GPU is available.

## Dataset

The model expects a dataset with:
- A folder containing `.jpg` images.
- A CSV file (`bbox.csv`) with columns: `file_name`, `xmin`, `xmax`, `ymin`, `ymax`, `object`.

Example CSV format:
```
file_name,xmin,xmax,ymin,ymax,object
image1.jpg,50,150,60,160,cat
image2.jpg,30,120,40,140,dog
...
```

Place the dataset in a directory (e.g., `data/`) and update the `folder_path` in the `CustDat` class if needed.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/object-detection-unet-yolo.git
   cd object-detection-unet-yolo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset by placing images and the CSV file in the `data/` directory.

## Usage

1. Open the `object_detection_notebook.ipynb` in Jupyter Notebook or JupyterLab.
2. Update the `dataset_path` variable to point to your dataset directory if different from `/kaggle/input/project-dataset`.
3. Run the notebook cells sequentially to:
   - Load and preprocess the dataset.
   - Initialize and train the model.
   - Visualize predictions on a test image.
   - Plot the training loss curve.

To train the model:
```python
# Key hyperparameters
num_epochs = 60
batch_size = 16
lr = 0.001
S = 7  # Grid size
```

The trained models are saved as `unet_model.pth` and `classify_model.pth`.

## Model Architecture

- **Net (U-Net)**: A feature extractor with four convolutional blocks in the downsampling path and four transposed convolutional blocks in the upsampling path, using skip connections to preserve spatial information.
- **Classify**: A classification head with two convolutional layers and average pooling, outputting a 7x7 grid with 49 channels (20 class probabilities, 1 object confidence, 4 bounding box coordinates per cell).

The loss function combines:
- **Classification Loss**: Cross-Entropy Loss for class predictions.
- **Localization Loss**: Mean Squared Error for bounding box coordinates (x, y, width, height).
- **Objectness Loss**: Mean Squared Error for object confidence scores.

## Results

The model was trained for 60 epochs, with the loss decreasing steadily as shown in the loss vs. epoch plot. The final cell in the notebook demonstrates how to visualize a predicted bounding box on a test image.

Example output:
- Predicted Class: [Class Name]
- Confidence: [Confidence Score]
- Bounding Box: (x=[x], y=[y], w=[width], h=[height])

## References

This work is published on the paper:
- Sharma, A. (2025). *Advancements in Object Detection Algorithms: Evaluating, Enhancing and Envisioning*. Vellore Institute of Technology.

Additional references:
1. Bochkovskiy, A., et al. "YOLOv4: Optimal Speed and Accuracy of Object Detection." ECCV, 2020.
2. He, K., et al. "Mask R-CNN." ICCV, 2017.
3. Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI, 2015.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for suggestions or bug reports.
