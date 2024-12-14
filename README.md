# Project Overview

This repository documents my work on two tasks: **Image Processing** and **Text Processing**, each contributing equally to the overall project (50% each). Below is a detailed explanation of what I did for each task.

## Task 1: Image Processing (50%)

I tackled the problem of weed classification using image processing techniques. Specifically, I focused on classifying two weed classes: **charlock** and **cleavers**, to simplify the problem. Here’s what I did:

1. **Dataset Overview**:
   - The dataset, adapted from a Kaggle competition, is structured into three folders:
     - **train**: Used to train the model.
     - **validation**: Used to tune hyperparameters.
     - **test**: Used exclusively for evaluating the trained model.
   - The class labels are embedded in the file names.

2. **Approach**:
   - **Feature Extraction**: I explored multiple visualizations and feature extractors to represent the images effectively for classification.
   - **Classifier**: I implemented machine learning classifiers to perform the weed classification task. Libraries like `scikit-learn` and `TensorFlow` were used to experiment with different models.
   - **Performance Evaluation**: I evaluated the classifiers using metrics such as accuracy, precision, recall, and F1-score. Visual representations like confusion matrices were generated to compare predictions against ground truth.

3. **Additional Contributions**:
   - I tested both custom implementations and library-based models to compare performance.
   - Resources used were acknowledged in the report, adhering to the project guidelines.

4. **Important Notes**:
   - The test dataset was strictly reserved for final evaluation and was not involved during training.

## Task 2: Text Processing (50%)

I performed topic modeling on a dataset containing documents from 5 domains. Here’s a breakdown of my work:

1. **Dataset Overview**:
   - The dataset comprises documents categorized into 5 domains:
     - **business**, **entertainment**, **politics**, **sport**, **tech**.
   - Each domain has a different number of documents, stored in separate subfolders.

2. **Approach**:
   - **Data Splitting**: I split the dataset into training (70%) and testing (30%) subsets.
   - **Preprocessing**: I implemented text preprocessing techniques such as tokenization, stopword removal, and stemming/lemmatization.
   - **Topic Labeller Versions**:
     - Version 1: Used a machine learning-based approach (e.g., `Naive Bayes`, `SVM`).
     - Version 2: Implemented a rule-based approach with custom preprocessing steps.
   - **Feature Extraction**: Employed techniques like Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF).

3. **Evaluation**:
   - The performance of both versions was compared using metrics like accuracy and confusion matrices.
   - Visualizations were included to highlight the differences between predicted labels and ground truth.

4. **Important Notes**:
   - I ensured that the evaluation metrics were generated using the test dataset only.

## Submission Checklist

- All necessary cells were executed to validate the solutions.
- Partial or erroneous code was commented out to prevent breaking the workflow.
- I acknowledged all external resources used and provided detailed explanations where required.

## Tools and Libraries Used

- **Image Processing**: OpenCV, scikit-image, TensorFlow.
- **Text Processing**: NLTK, scikit-learn, Gensim.

---

Happy coding!
