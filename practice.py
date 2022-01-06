# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

# load data
full_df = pd.read_csv("C:/Users/Documents/Projects/Datasets/sales_data_sample.csv")
# smaller dataframe with variables of interest
df = pd.DataFrame(full_df, columns=['ORDERNUMBER', 'QUANTITYORDERED', 'PRICEEACH', 'ORDERDATE', 'QTR_ID', 'PRODUCTLINE', 'PRODUCTCODE', 'MSRP'])
# rename the columns
df.rename(columns={'ORDERNUMBER': 'order_number', 'QUANTITYORDERED': 'quantity', 'PRICEEACH': 'price', 'ORDERDATE': 'order_date', 'QTR_ID': 'quarter', 'PRODUCTLINE': 'product_line', 'PRODUCTCODE': 'product_code'}, inplace=True)
# reorder the dataframe by date
df['datet'] = pd.to_datetime(df['order_date'])
df = df.sort_values(by='datet').set_index('datet')


#####


# Get relative difference between sales of two months
# *question was get relative difference between sales of two years


# add revenue column (revenue = quantity*price)
df['revenue'] = df['quantity'] * df['price']

# convert order_date to datetime
df['order_date'] = pd.to_datetime(df['order_date'])

# add a year column and a month column
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month

# get total quantities ordered from July 2003 and August 2003
sales_jul03 = df[(df['month'] == 7) & (df['year'] == 2003)]['revenue'].sum()
sales_aug03 = df[(df['month'] == 8) & (df['year'] == 2003)]['revenue'].sum()

# calculate relative difference using formula: (month2 / month1 - 1)
rel_diff = (sales_aug03 / sales_jul03 - 1)
print('Relative difference between July 2003 sales and August 2003 sales is: $', round(rel_diff, 2))


#####


# Get median quantity ordered (essentially the median order size) by product line
# *question was get median age by aquisition channel 


df.product_line.unique() # see what the product lines are

# group by product line, get the median
print(pd.DataFrame(df.groupby(['product_line'])['quantity'].median()))


#####


# Most popular product in March
# *question was which product was most popular among people born in 19xx-19xx


# subset for March of every year
mar = df[(df['month'] == 3)]

# get the mode product code
pop_prod = mar['product_code'].value_counts().idxmax()
pop_line = mar['product_line'].value_counts().idxmax()
print('The most popular product sold in the month of March is', pop_prod, 'from the', pop_line, 'product line.')


#####


# Regression question
# dependent variable: revenue aggregated at the monthly level


# create an aggregated table of revenue by month-year
df['ym'] = df['order_date'].dt.strftime('%m %Y')
aggtab = pd.DataFrame(df.groupby(pd.Grouper(key='ym')).agg(revenue=('revenue','sum'), unique=('product_code','nunique'), avg_msrp=('MSRP','mean'), units=('quantity','sum')))

# add features: (omitted age because of limited data)
# prev month's revenue
aggtab['prev_rev'] = aggtab['revenue'].shift(+1)
# prev month's number of distinct products
aggtab['prev_distinct'] = aggtab['unique'].shift(+1)

# additional features: 
# average MSRP, total quantity of product sold in the month added to aggtab earlier

# create training and testing sets. test = Jan-May 2005, train = rest of data
aggtab = aggtab.reset_index()
train = aggtab.iloc[1:-5] 
test = aggtab.iloc[-5:]
train_x = train.drop(['ym','revenue'],axis=1)
train_y = train.loc[:,'revenue']
test_x = test.drop(['ym','revenue'],axis=1)

# fit model and predict for test set
model = LinearRegression().fit(train_x, train_y)
predicted = model.predict(test_x)

# compute RMSE
actual_col = test.loc[:,'revenue']
actual = actual_col.values
rmse = sqrt(mean_squared_error(actual, predicted)) # not bad, considering the values in our data










