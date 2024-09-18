# Image Classification Using ResNet50 and Vision Transformer (ViT)

This repository implements an image classification pipeline using pre-trained **ResNet50** and **Vision Transformer (ViT)** models to extract features from images. These features are used to make predictions with a traditional machine learning classifier.

## Features

- **ResNet50**: Extracts convolutional-based image features.
- **Vision Transformer (ViT)**: Provides transformer-based image features.
- **Combined Feature Extraction**: Merges ResNet50 and ViT features for enhanced classification performance.
- **Pre-trained ML Model**: Uses a pre-trained model for classification.
- **Efficient Image Downloading**: Downloads images directly from URLs for processing.

## Approach

1. **Image Download**: Images are downloaded from the given URLs using the `requests` library and processed using the `PIL` library to convert them to RGB.
2. **Feature Extraction**:
   - **ResNet50**: Pre-trained ResNet50 extracts high-quality features using convolutional layers.
   - **Vision Transformer (ViT)**: A transformer model extracts features using attention mechanisms, providing a complementary view of the image's representation.
3. **Feature Combination**: Features from both models are concatenated to form a single feature vector.
4. **Prediction**: The combined features are passed to a pre-trained machine learning model (using `joblib`) to classify the image.
5. **Prediction Storage**: The results are saved as a CSV file for analysis.

## Prerequisites

- **Python 3.8+**
- **PyTorch**
- **Transformers (Hugging Face)**
- **Torchvision**
- **Pandas**
- **Joblib**
- **TQDM**
- **Requests**
- **Pillow (PIL)**

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/image-classification-resnet50-vit.git
    cd image-classification-resnet50-vit
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare Dataset**:
    - Add the test data (image URLs, categories, and other details) in the file: `dataset/sample_test.csv`.
    - Ensure the test CSV file has the necessary columns: `image_link`, `group_id`, `entity_name`.

2. **Run the Model**:

    Execute the `main.py` script:

    ```bash
    python main.py
    ```

3. **Output**:
    - The predictions will be saved to a CSV file: `dataset/test_out.csv`.
