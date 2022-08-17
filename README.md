# Speech emotion recognition on RAVDESS

The goal of the project is to perform speech emotion recognition on audio files. 
The dataset in use contains both songs and speech, by 24 distinct actors performing 8 different emotions. 

## Code structure
All the code is organized inside `src` where you can find: 
- `src/loader.py`: loading of raw data; 
- `src/data_preprocessing.py`: feature extraction from raw data;
- `src/dataset.py`: dataset management, such as splits, scaling, etc;
- `src/models.py`: collection of functions that define or train models;
- `src/utility.py`: plots and initial study on the dataset.

## Notebooks

The study is divided among different notebooks: 
- `01_exploration.ipynb`: initial study on the dataset; 
- `02_training.ipynb`: initial training of neural networks and simpler models;
- `03_data_augmentation.ipynb`: experiments with data augmentation.
- `04_cross_validation.ipynb`: cross validation results.
    
[Report link](https://github.com/tomfran/ravdess-ser/blob/main/report/report.pdf)
