# Face Mask Detector

Detecting face masks in real-time using deep learning and computer vision techniques.

## Table of Contents
- [Project Motto](#project-motto)
- [Project Description](#project-description)
- [Methodologies](#methodologies)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Achievements](#achievements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Motto
"Ensuring Safety Through Technology"

## Project Description
The Face Mask Detector project aims to develop a robust system for detecting whether individuals are wearing face masks in real-time. This project is crucial in the current global scenario where wearing face masks has become a necessity to curb the spread of infectious diseases.

Using deep learning and computer vision, the project utilizes a Convolutional Neural Network (CNN) to classify images of people into two categories: with mask and without mask. The model is trained on a diverse dataset to ensure accuracy and reliability.

## Methodologies
The project follows a structured approach comprising the following methodologies:

1. **Data Collection**:
   - A diverse dataset consisting of images of people with and without masks was collected from various sources.
   - The dataset was annotated to label each image as 'with_mask' or 'without_mask'.

2. **Data Preprocessing**:
   - Images were resized to a standard dimension to ensure uniformity.
   - Data augmentation techniques such as rotation, zoom, and flip were applied to increase the dataset size and variability.

3. **Model Architecture**:
   - A Convolutional Neural Network (CNN) was designed to extract features from the images.
   - The model architecture includes several convolutional layers followed by pooling layers, and fully connected layers leading to the final output layer.

4. **Model Training**:
   - The dataset was split into training and validation sets.
   - The model was trained using the training set, and its performance was evaluated on the validation set.
   - Hyperparameters such as learning rate, batch size, and number of epochs were fine-tuned to optimize the model's performance.

5. **Model Evaluation**:
   - The model was evaluated using metrics such as accuracy, precision, recall, and F1-score to assess its performance.
   - A confusion matrix was plotted to visualize the classification results.

6. **Real-Time Detection**:
   - The trained model was integrated with OpenCV to enable real-time face mask detection using a webcam or video feed.

## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/MadanMohanBammidi/Face-Mask-Detector.git
    cd Face-Mask-Detector
    ```

## Usage
1. **Run the main script**:
    ```sh
    python main.py
    ```

2. **Provide a video feed**:
   - Ensure your webcam is connected, or use a video file as input.
   - The script will process the video feed and display the results in real-time.

## Features
- Real-time face mask detection
- High accuracy and reliability
- Easy integration with existing systems
- Robust against variations in lighting and background

## Achievements
- **Accuracy**: Achieved high accuracy in detecting face masks in real-time.
- **Real-Time Performance**: Successfully integrated the model with OpenCV for real-time detection.
- **Scalability**: Developed a scalable solution that can be deployed in various environments such as offices, public transport, and retail stores.

## Contributing
Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

1. **Fork the repository**
2. **Create a new branch**:
    ```sh
    git checkout -b feature-branch
    ```
3. **Make your changes**
4. **Commit your changes**:
    ```sh
    git commit -m "Add some feature"
    ```
5. **Push to the branch**:
    ```sh
    git push origin feature-branch
    ```
6. **Open a pull request**

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or inquiries, please contact [Madan Mohan Bammidi](https://github.com/MadanMohanBammidi).
