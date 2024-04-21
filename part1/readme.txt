Part 1: Readme

Wine-Dataset k-Nearest Neighbour algorithim

This program implements the k Nearest Neighbour (KNN) algorithim,
which uses training data to train and make predictions on the test data. 
The class to be predicted is identified as "class" and is the last column in the train/data csv files
The other 13 columns are attributes of the Wine dataset. 

Prerequesits
- python3
- numpy
- pandas
- Train and test set found in the /part1 directory 

Usage
1. Navigate to the /part1 folder containing knn.py
2. Run the program using the following command  
    python3 knn.py wine_train.csv wine_test.csv output k
3. Replace k with your preferred k value 
4. Once the program has been run, the Accuracy will be printed to the terminal 
    and the ouput file will be saved to /part2

