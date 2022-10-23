# Big assignment for machine learning
Improved K-nearest neighbor algorithm based on Gaussian function for letter recognition.
## Requirements
+ OpenCV >= 3.0
+ Python >= 3.6
## Dataset
+ Download FROM: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz
+ Initialize the dataset
```
  #python3 initial.py
 ```
## Usage
+ If you want to change the data set, use `getBest_K()` in `main`.
+ To get the test set, you also need to use `get_test_img` in `main`.
+ Now you can use `predict` to get results.
```
  #python3 main.py
  ```
## Sample Results
The image of handwritten letter A is obtained by CV and then recognized
![Demo1](https://github.com/Kimo1116/ml/blob/main/test1.jpg)
The results are as follows：
```
前十个候选为 ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'G', 'A', 'A']
预测结果为  A
```
