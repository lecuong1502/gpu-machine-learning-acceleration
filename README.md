COVID-19 Radiography Image Dataset Processing

1. Environment:
    - PyTorch, CUDA.
    - Libraries: torch, torchvision, torchaudio, scikit-learn, matplotlib, opencv-python, numpy, torchsummary

2. Data and processing the dataset:
    - I download and use COVID-19_Radiography_Dataset from Kaggle: https://www.kaggle.com/datasets/preetviradiya/covid19-radiography-dataset. This dataset is from a team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. This COVID-19, normal and other lung infection dataset is released in stages. In the first release they have released 219 COVID-19, 1341 normal and 1345 viral pneumonia chest X-ray (CXR) images. In the first update, they have increased the COVID-19 class to 1200 CXR images. In the 2nd update, they have increased the database to 3616 COVID-19 positive cases along with 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection) and 1345 Viral Pneumonia images.
    - DataLoader: I used PyTorch DataLoader to download data batches.
    - I split the data into 3 parts: training, validation, testing to evaluate the model.

    ![alt text](/image/data_processing.png)

3. Build the model and Train the Unet model
    - UNet is a common convolutional neural network architecture for image segmentation. UNet includes contracting path (view) and expanding path (positioning)
    
4. Evaluating the model:
    - Accuracy: Ratio of predicted errors to total.
    - Confusion Matrix: Shows the number of correctly and incorrectly classified samples for each class.
    - Recall: Rate of correct prediction of positive COVID-19 cases
    - Specificity: Rate of correct prediction of negative COVID-19 cases

5. CUDA application:
    - .to(device): Move the model and data (images, labels) to the GPU. This is the step to speed up the computation.
    - num_workers: Set num_workers > 0 in DataLoader to load data from disk in parallel, keeping GPU from waiting.
    - GPU:
    ![alt text](/image/GPU.png)