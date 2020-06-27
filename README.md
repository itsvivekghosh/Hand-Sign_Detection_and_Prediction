# Hand-Sign Detection and Prediction

## Table of Content:
#### 1. Cloning the Repository
#### 2. Installation of Our Requirements
#### 3. Testing Model ***(Optional)***
#### 4. Prediction
#### 5. Results:

### File Structure:
```
D:.
├───data
│   ├───test
│   │   ├───0
│   │   ├───1
│   │   ├───2
│   │   ├───3
│   │   ├───4
│   │   └───5
│   └───train
│       ├───0
│       ├───1
│       ├───2
│       ├───3
│       ├───4
│       └───5
├───data_collection
│   └───__pycache__
├───data_modeling
│   ├───testing
│   │   └───__pycache__
│   ├───training
│   │   └───__pycache__
│   └───__pycache__
├───model
├───outputs
└───prediction
    └───__pycache__
```

## 1. Cloning the Repository:

You can clone this repository in your directory by entering the command:
```
$ git clone <repo> <directory>
```

In our case use this command:
```
$ git clone git://github.com/itsvivekghosh/Hand-Sign_Detection_and_Prediction.git
```

## 2. Installation of Our Requirements:

All the requirement libraries are added into `requirements.txt` file:

Your can install all the libraries by just entering the below command into your bash or command prompt:
- For Python < 3:
```
pip install -r requirements.txt
``` 
- For Python >= 3:
```
pip3 install -r requirements.txt
```

## 3. Testing The Model: ***(Optional)***

Its very easy to test the model. But, we need to follow some steps to that:
- Open `main.py` file which is the main directory.
- After Opening this file, uncomment __`line 27`__ where the line is __`testModel(...)`__ . 
- Now, run the `main.py` file for testing the model is working properly or not.

***None:- After successfull testing of the Model, the camera will Open immediately next step (Prediction) for solving this, uncomment the __`line 37`__ and then go for the further steps..***

## 4. Prediction: 

After Successfull Testing (if implemented last step) of the Model, run `main.py` file using this simple command:
- For Python < 3:
```
python main.py
```
- For Python >= 3:
```
python3 main.py
```
***Note: Please make sure that __`main.py`__ file has __`line 37`__ uncommented out otherwise will generate errors.***

## 5. Results:
After The Predictions from WebCam, the result can be something like these:<br><br>
<img src="https://github.com/itsvivekghosh/Hand-Sign_Detection_and_Prediction/blob/master/Outputs/ZERO.png" width=300px height=300px></img>
<img src="https://github.com/itsvivekghosh/Hand-Sign_Detection_and_Prediction/blob/master/Outputs/ONE.png" width=300px height=300px></img>
<img src="https://github.com/itsvivekghosh/Hand-Sign_Detection_and_Prediction/blob/master/Outputs/TWO.png" width=300px height=300px></img>
<img src="https://github.com/itsvivekghosh/Hand-Sign_Detection_and_Prediction/blob/master/Outputs/THREE.png" width=300px height=300px></img>
<img src="https://github.com/itsvivekghosh/Hand-Sign_Detection_and_Prediction/blob/master/Outputs/FOUR.png" width=300px height=300px></img>
<img src="https://github.com/itsvivekghosh/Hand-Sign_Detection_and_Prediction/blob/master/Outputs/FIVE.png" width=300px height=300px></img>


## About Author:
### Code Written by:
__Vivek Kumar Ghosh__<br>
__Uttaranchal University, Dehradun__<br>
__Bachelor of Technology, Computer Science__<br>
__E-mail: soapmactevis1@gmail.com__<br>
__LinkedIn: https://www.linkedin.com/in/itsvivekghosh__<br>

#### ***For any bugs and queries feel free to contact the above profiles. Well, we are working on fixing other Bugs.***
# Happy Coding!
