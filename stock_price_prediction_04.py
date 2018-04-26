import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#plt.switch_backend('newbackend')

dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)

        i = 1

        for row in csvFileReader:
            #dates.append(int (row[0].split('-')[0] + row[0].split('-')[1] + row[0].split('-')[2]))
            dates.append(int (i))
            prices.append(float(row[1]))
            i = i + 1

    print('reading done!')
    #print(dates)
    #print(prices)
    return

def predict_prices(dates, prices, x):
    print('starting prediction...')
    dates = np.reshape(dates, (len(dates), 1))

    svr_lin = SVR(kernel = 'linear', C = 1e3)
    svr_poly = SVR(kernel = 'poly', C = 1e3, degree = 2)
    svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)

    print('starting linear model')
    svr_lin.fit(dates, prices)
    print('linear model, done!')


    print('starting Polynomial model')
    svr_poly.fit(dates, prices)
    print('Polynomial model, done!')

    print('starting RBF model')
    svr_rbf.fit(dates, prices)
    print('RBF model, done!')

    plt.scatter(dates, prices, color = 'black', label = 'Data')

    plt.plot(dates, svr_rbf.predict(dates), color = 'red', label = 'RBF model')
    plt.plot(dates, svr_lin.predict(dates), color = 'green', label = 'linear model')
    plt.plot(dates, svr_poly.predict(dates), color = 'blue', label = 'Polynomial model')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()

    print('done!')
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('C:/Users/philipp.s/Desktop/SR_LPfDS/AAPL_m.csv')

#day = []
#ppa = []
#day.append(26)

print(dates)
day = len(dates) + 1
print("predicting day: %s" % day)

predicted_price = predict_prices(dates, prices, day)
#ppa.append(predicted_price)



#plt.plot(day, ppa, color = 'red', label = 'predicted')
#plt.show()

print(predicted_price)
