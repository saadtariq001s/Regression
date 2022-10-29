import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

path = r'C:\Users\saad\Desktop\Faces\exams.csv'
student = pd.read_csv(path)
students = student[["writing_score", "reading_score"]]
# print(students)

 #[gender, race/ethnicity, parental level of education, lunch, test preparation course, math_score, reading_score, writing_score]

# plt.scatter(student.writing_score, student.math_score, student.reading_score)
# plt.ylabel("Math Score")
# plt.xlabel("Reading/Writing")
# plt.show()

msk = np.random.rand(len(student)) < 0.8
train = students[msk]
test = students[~msk]

train_x = np.asanyarray(train[["writing_score"]])
train_y = np.asanyarray(train[["reading_score"]])

test_x = np.asanyarray(test[["writing_score"]])
test_y = np.asanyarray(test[["reading_score"]])

poly = PolynomialFeatures(degree= 2)

poly_train = poly.fit_transform(train_x)


regr = linear_model.LinearRegression()

model = regr.fit(poly_train, train_y)


print("Coeff:", regr.coef_)
print("Intercept:", regr.intercept_)
print("Mean Squared Error", mean_squared_error(test_x, test_y))
print("Variance:", r2_score(test_x, test_y))

plt.scatter(train.writing_score, train.reading_score,  color='blue')
XX = np.arange(0.0, 100.0, 2.0)
yy = regr.intercept_[0]+ regr.coef_[0][1]*XX+ regr.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Writing Score")
plt.ylabel("Reading Score")

# plt.scatter(test_x, test_y, color='blue')
# plt.plot(test_x, predicted_score)

plt.show()


