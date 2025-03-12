import numpy as np
import csv

file_name = "./Advertising.csv"

# Initialize numpy arrays
tv, radio, newsp, sales = np.array([]), np.array([]), np.array([]), np.array([])
# Read the CSV file and extract data
with open(file_name, mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  #Skip header
    for row in csv_reader:
        tv = np.append(tv, float(row[1]))
        radio = np.append(radio, float(row[2]))
        newsp = np.append(newsp, float(row[3]))
        sales = np.append(sales, float(row[4]))

assert len(tv) == len(radio) == len(newsp) == len(sales)
length = len(tv)

#Linear regression
def linear_regression(x):
    x_mean, sales_mean = np.mean(x), np.mean(sales)  #Compute means of independent and dependent variables
    #Compute slope (beta1) using the least squares formula
    beta1 = round(np.sum((x - x_mean) * (sales - sales_mean)) / np.sum((x - x_mean) ** 2), 5)
    #Compute intercept (beta0)
    beta0 = round(sales_mean - beta1 * x_mean, 5)   
    #Predict sales using the linear regression equation
    sales_pred = beta0 + beta1 * x    
    #Compute the Residual Sum of Squares (RSS)-measures error
    RSS = np.sum((sales - sales_pred) ** 2)
    #Compute Residual Standard Error (RSE)-standard deviation of residuals
    RSE = np.sqrt(RSS / (length - 2))
    #Compute Total Sum of Squares (TSS)-measures total variation in sales
    TSS = np.sum((sales - sales_mean) ** 2)
    #Compute R-squared (R²)-proportion of variance explained by the model
    R2 = (TSS - RSS) / TSS  
    #Compute F-statistic-measures overall significance of the model
    FS = ((TSS - RSS) / 1) / (RSS / (length - 2))  
    #Print regression results
    print("Slope:", beta1, "Intercept:", beta0)
    print("RSS:", RSS, "RSE:", RSE, "R2:", R2, "F-Statistic:", FS)

#Run regression
    print("TV Advertising:")
    linear_regression(tv)
    print("\nRadio Advertising:")
    linear_regression(radio)
    print("\nNewspaper Advertising:")
    linear_regression(newsp)
