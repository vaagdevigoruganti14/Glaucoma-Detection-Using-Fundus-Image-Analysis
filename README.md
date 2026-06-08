Glaucoma-Detection-Using-Fundus-Image-Analysis

Description
This project is a web-based glaucoma detection system that uses Deep Learning to analyze retinal fundus images and identify whether glaucoma is present or not. The system uses the VGG16 model for image classification and provides results through a simple Flask web application.

Technologies Used:

Python
TensorFlow
Keras
Flask
HTML
CSS
Bootstrap
NumPy

Project Files:

- app.py – Flask application
- Train.py – Model training code
- fronted.html – User interface
- Screenshots – Output screenshots
- README.md – Project documentation

How to Run:

1. Install required libraries.
2. Run the training script.
3. Run the Flask application.

Note: We collected glaucoma and normal fundus images from publicly available datasets. The images were organized into training and validation folders. We used VGG16 transfer learning to extract retinal features and trained a binary classification model. After training, the model was saved as an H5 file. A Flask web application was then developed to allow users to upload fundus images. The uploaded image is preprocessed, passed through the trained model, and the prediction result is displayed as either Glaucoma Detected or Normal Eye
