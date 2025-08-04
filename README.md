# Tree Species Classification using Deep Learning 🌳

This project uses a Convolutional Neural Network (CNN) with transfer learning to classify over 100 different species of trees from their images. The model is built using TensorFlow and Keras and is designed to run in a Google Colab environment.

## 🚀 Overview

- **Model:** MobileNetV2 (pre-trained on ImageNet)
- **Framework:** TensorFlow / Keras
- **Dataset:** [100+ Tree Species Dataset from Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/100-plus-tree-species-dataset-for-image-classification)
- **Environment:** Google Colab / Jupyter Notebook

---
## 🛠️ How to Run This Project

1.  **Download the Dataset:** Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/gauravduttakiit/100-plus-tree-species-dataset-for-image-classification) and download the dataset zip file to your computer. It will likely be named `archive.zip`.

2.  **Open in Google Colab:** Open the `Tree_Species_Classification.ipynb` notebook in Google Colab.

3.  **Upload the Dataset:**
    - In the Colab notebook, click on the folder icon 📁 on the left sidebar.
    - Click the upload button (📄⬆️) and upload the `archive.zip` file you downloaded. This may take some time.

4.  **Run the Cells:**
    - Run the first code cell to unzip the dataset.
    - Once the first cell is finished, run the second code cell to train the model.

5.  **Done!** The model will train and save itself as `tree_species_classifier.h5`.

---
## 📈 Result

The model was trained for 10 epochs and achieved a good accuracy on the validation set, demonstrating the effectiveness of transfer learning for image classification tasks.
