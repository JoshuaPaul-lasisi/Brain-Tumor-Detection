# **Brain Tumor Detection using Deep Learning**

## **Overview**  
This project focuses on detecting brain tumors from MRI images using deep learning techniques. A Convolutional Neural Network (CNN) was developed to classify MRI images into two categories: **tumor** and **no tumor**. The solution includes a web-based interface powered by Flask, enabling users to upload images and receive real-time predictions.

---

## **Table of Contents**
1. [Project Features](#project-features)  
2. [Technologies Used](#technologies-used)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Dataset](#dataset)  
6. [Model Training](#model-training)  
7. [Results](#results)  
8. [Future Improvements](#future-improvements)  
9. [Contributing](#contributing)  
10. [License](#license)  

---

## **Project Features**  
- **Data Preprocessing:** Resizing, normalization, and splitting of MRI image datasets.  
- **Deep Learning Model:** CNN-based model trained using TensorFlow and Keras for high accuracy in image classification.  
- **Web Interface:** User-friendly Flask application for uploading images and viewing predictions.  
- **Model Evaluation:** Assessed using accuracy, precision, recall, and F1 score metrics.  
- **Deployment:** Local deployment with Flask for real-time predictions.  

---

## **Technologies Used**  
- **Programming Languages:** Python  
- **Libraries and Frameworks:** TensorFlow, Keras, Flask, OpenCV, NumPy, Matplotlib  
- **Web Development:** HTML, CSS, JavaScript  
- **Machine Learning Concepts:** Convolutional Neural Networks, Deep Learning, Image Classification  

---

## **Installation**  

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/yourusername/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. **Set Up a Virtual Environment:**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**  
   ```bash
   python flask_app.py
   ```

5. **Access the Application:**  
   Open your browser and go to `http://127.0.0.1:5000`.

---

## **Usage**  

1. **Upload Image:** Navigate to the main page and upload an MRI image.  
2. **Get Prediction:** Click the "Predict" button to view the classification result.  

---

## **Dataset**  
The dataset contains labeled MRI images of brain scans categorized as **tumor** and **no tumor**. Data was preprocessed to ensure optimal performance during training and testing. [Dataset source](https://www.kaggle.com/...)

---

## **Model Training**  

- **Architecture:** Convolutional Neural Network (CNN) with multiple convolutional, pooling, and dense layers.  
- **Training Details:**  
  - Optimizer: Adam  
  - Loss Function: Binary Cross-Entropy  
  - Epochs: 25  
  - Metrics: Accuracy, Precision, Recall  

---

## **Results**  

- **Model Performance:**  
  - Training Accuracy: 95%  
  - Validation Accuracy: 92%  

- **Evaluation Metrics:**  
  - Precision: 94%  
  - Recall: 91%  
  - F1 Score: 92.5%  

---

## **Future Improvements**  

1. Deploy the application using cloud services (e.g., AWS, Google Cloud).  
2. Optimize the model for faster inference on large datasets.  
3. Integrate additional data augmentation techniques to improve generalization.  
4. Expand the web interface with enhanced UX/UI features.  

---

## **Contributing**  

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and create a pull request.  

1. Fork the project  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m 'Add some feature'`)  
4. Push to the branch (`git push origin feature/YourFeature`)  
5. Open a Pull Request  

---

## **License**  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  

