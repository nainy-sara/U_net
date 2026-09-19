# U-Net for Pet Image Segmentation

This project implements a **U-Net** model using TensorFlow and Keras for semantic segmentation on the [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset.

The model receives an image of a pet and predicts one of the following three classes for each pixel:

- Background
- Pet boundary
- Pet

## Features

- Automatically downloads the dataset through TensorFlow Datasets
- Resizes images and masks to `128×128`
- Normalizes images to the `[0, 1]` range
- Applies random horizontal flipping for data augmentation
- Uses the U-Net encoder–decoder architecture
- Trains with the `sparse_categorical_crossentropy` loss function
- Displays the input image, ground-truth mask, and predicted mask
- Saves output images to `output.png`

## Project Structure

The main project file is currently:

```text
.
└── U_net2.py
```

## Requirements

To run this project, you need Python 3.8 or later and the following packages:

- TensorFlow
- TensorFlow Datasets
- NumPy
- Matplotlib

Install the dependencies with:

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib
```

> To use a GPU, install a TensorFlow version compatible with your system and CUDA drivers.

## Usage

Clone the repository and move into the project directory:

```bash
git clone https://github.com/nainy-sara/U_net.git
cd U_net
```

Then run the script:

```bash
python U_net2.py
```

On the first run, the `oxford_iiit_pet:4.0.0` dataset will be downloaded automatically. An internet connection and sufficient storage space are therefore required.

## Data Processing Pipeline

### Training

The following steps are applied to the training data:

1. Load the image and segmentation mask
2. Resize them to `128×128`
3. Apply a random horizontal flip
4. Convert image pixel values to `float32`
5. Normalize image values by dividing them by `255`
6. Convert the original mask labels to classes from `0` to `2`

### Validation and Testing

The test data is processed without data augmentation. Part of the test data is used for validation, while another part is reserved for testing.

## U-Net Architecture

The model consists of the following components:

- **Encoder:** Four downsampling blocks with `64`, `128`, `256`, and `512` filters
- **Bottleneck:** Two convolutional layers with `1024` filters
- **Decoder:** Four upsampling blocks with skip connections
- **Output:** A convolutional layer with three channels and a `softmax` activation function

Overall model structure:

```text
Input (128, 128, 3)
        │
     Encoder
        │
   Bottleneck
        │
     Decoder
        │
Output (128, 128, 3)
```

## Training Configuration

The current training settings in `U_net2.py` are:

- Batch size: `64`
- Number of epochs: `20`
- Optimizer: `Adam`
- Loss function: `sparse_categorical_crossentropy`
- Evaluation metric: `accuracy`
- Image size: `128×128`

To change these values, edit the `BATCH_SIZE` and `NUM_EPOCHS` variables in the main file.

## Outputs

During execution:

- The model architecture summary is displayed in the terminal.
- A sample image containing the input image and ground-truth mask is saved to:

```text
output.png
```

The `show_predictions` function is also defined in the code for displaying model predictions on test data.

## Important Notes

- The model is trained from scratch each time the script is run, and its weights are not saved automatically.
- To continue training or reuse the model, you can use `model.save()` and `load_model()`.
- The `validation_batches` dataset is created from the test data. For a standard evaluation workflow, it is recommended to create a separate validation dataset.
- Before using the model in production, consider adding metrics such as IoU, Dice Score, and Mean IoU.
- Training this model with a batch size of `64` and a large number of filters may require a considerable amount of GPU memory. If you run out of memory, reduce the `BATCH_SIZE` value.

## Suggested Improvements

- Save the best model weights using `ModelCheckpoint`
- Use `EarlyStopping`
- Add Dice and IoU metrics
- Apply additional data augmentation, such as rotation and brightness adjustment
- Separate the training, validation, and test datasets
- Save loss and accuracy plots
- Use mixed precision for faster training on compatible GPUs

## License

This repository does not currently include a specific license file. Before using or redistributing the code, review the usage terms of the Oxford-IIIT Pet dataset and the license of this project.
