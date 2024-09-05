# Osteoarthritis and Image Classification Project

This project focuses on image classification using deep learning techniques to differentiate between classes of images related to osteoarthritis and other datasets (such as cats and dogs). The project leverages TensorFlow, Keras, and transfer learning models like InceptionV3 and InceptionResNetV2 to achieve high accuracy.

## Project Structure

- **Train Dataset**: Contains the training images.
- **Validation Dataset**: Contains images used for model validation.
- **Test Dataset**: Contains test images for evaluating model performance.

## Datasets

The datasets used are:
1. **Osteoarthritis Dataset**: Used to classify medical images.
2. **Cats and Dogs Dataset**: A binary classification problem to distinguish between images of cats and dogs.

## Model Overview

This project uses two pre-trained models for transfer learning:
1. **InceptionV3**: Fine-tuned for osteoarthritis image classification.
2. **InceptionResNetV2**: Fine-tuned for cats and dogs classification.

### Transfer Learning

We use pre-trained models (`InceptionV3` and `InceptionResNetV2`) from Keras with ImageNet weights. The top layers were removed and custom dense layers were added for binary classification.

### Data Augmentation

Data augmentation was applied to increase the variability of the training set and improve the generalization capability of the model:
- **Rotation**: Up to 40 degrees.
- **Width and Height Shifts**: Random shifts up to 20%.
- **Shear and Zoom Transformations**: Up to 20%.
- **Horizontal Flip**: Random flips to simulate different perspectives.

## Training Process

For both datasets:
- **Optimizer**: RMSprop (learning rate = 2e-5)
- **Loss Function**: Binary Crossentropy (used for binary classification).
- **Metrics**: Accuracy is used to monitor performance.

The models were trained for 10 epochs with augmented data. The training and validation accuracy and loss are plotted to visualize performance over time.

## Results

### Osteoarthritis Classification (InceptionV3):
- Achieved a validation accuracy of 91.5% after 10 epochs.
- Training accuracy: ~81.4%.

### Cats and Dogs Classification (InceptionResNetV2):
- Achieved a validation accuracy of 97.5% after 10 epochs.
- Training accuracy: ~95.6%.

## Visualization

Training history is plotted for both models, showing the trends in accuracy and loss during training.

## How to Run

1. Install required libraries:
    ```bash
    pip install tensorflow pandas numpy matplotlib
    ```
2. Place the datasets in the respective directories.
3. Run the training script to train the model:
    ```bash
    python train_model.py
    ```
4. Evaluate the model using the test dataset:
    ```bash
    python evaluate_model.py
    ```

## Future Improvements

- Tuning hyperparameters for better performance.
- Implementing additional techniques to reduce overfitting.
- Experimenting with different models and architectures for improved classification.

## License

This project is licensed under the MIT License.t.
