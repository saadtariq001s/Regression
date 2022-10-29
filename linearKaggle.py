import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

path = r'C:\Users\saad\Desktop\Faces\exams.csv'
student = pd.read_csv(path)
students = student[["writing_score", "reading_score","math_score"]]
# print(students)

 #[gender, race/ethnicity, parental level of education, lunch, test preparation course, math_score, reading_score, writing_score]

# plt.scatter(student.writing_score, student.math_score, student.reading_score)
# plt.ylabel("Math Score")
# plt.xlabel("Reading/Writing")
# plt.show()

msk = np.random.rand(len(student)) < 0.8
train = students[msk]
test = students[~msk]

train_x = np.asanyarray(train[["writing_score", "math_score"]])
train_y = np.asanyarray(train[["reading_score"]])

test_x = np.asanyarray(test[["writing_score", "math_score"]])
test_y = np.asanyarray(test[["reading_score"]])

regr = linear_model.LinearRegression()

regr.fit(train_x, train_y)

predicted_score = regr.predict(test_x)

print("Coeff:", regr.coef_)
print("Intercept:", regr.intercept_)
print("Mean Squared Error", mean_squared_error(test_x, test_y))
print("Variance:", r2_score(test_x, test_y))

plt.scatter(test_x, test_y, color='blue')
plt.plot(test_x, predicted_score, "-r")
plt.xlabel("Writing Score")
plt.ylabel("Reading Score")

plt.show()


