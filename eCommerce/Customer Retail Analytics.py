#!/usr/bin/env python
# coding: utf-8

# # Ecommerce Marketing Analytics
# 
# Submitter: Nikhil Sharma
# 
# ### Introduction
# 
# Companies are always trying to find ways to boost e-commerce sales by personalizing their marketing strategies to drive customer retention. In this project, a recommendation engine will be built using RFM (Recency, Frequency, Monetary) and cluster analysis to optimize customer lifetime value. The dataset that will be used is from the UCI Machine Learning Repository. This transactional dataset contains transactions occurring between December 1, 2010, and December 9, 2011, for an online retailer. Raw data includes 541,909 observations with 8 variables. The dataset includes 25,900 unique transactions with 3,958 unique products purchased by 4,372 different customers from 38 different countries.
# 
# ### Dataset Attributes
# 
# InvoiceNo (Nominal): Number assigned to each transaction;
# StockCode (Nominal): Product code;
# Description (Nominal): Product name;
# Quantity (Numeric): Number of products purchased for each transaction;
# InvoiceDate (Numeric): Timestamp for each transaction;
# UnitPrice (Numeric): Product price per unit;
# CustomerID (Nominal): Unique identifier each customer;
# Country (Nominal) Country name
# 
# ### Business Objective
# 
# To build a recommendation engine that will categorize customers in a particular segment of the population based on their buying patterns; discover customers that are expected to have higher returns; predict which products are frequently purchased, and increase customer retention and drive sales.
# 
# ### Questions
# 
# Who are the best customers?
# How are different products linked together by sales?
# Which customers are contributing to the churn rate? 
# Which customers are on the verge of churning?
# Who has the potential to become a valuable customer?
# Which customers can be retained?
# Which customers are more likely to respond to engagement campaigns?
# Who are your loyal customers?
# Who has the potential to be converted into a more profitable customer?
# Which group of customers is most likely to respond to your current campaign?

# # Importing Required Modules & Loading Dataset 

# In[2]:


# python libraries
import numpy as np # to perform mathematical operations on multidimensional matrices/arrays
import pandas as pd # to perform data manipulation/analysis
import matplotlib as mpl # to perform basic plotting
import matplotlib.pyplot as plt # to customize visualizations using command style functions
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly # interactive open-source plotting library
import plotly.offline as py # to make interactive visualizations online and save them offline if needed
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
init_notebook_mode() # to get the connection 
import plotly.graph_objs as go # to generate graph objects
from chart_studio.plotly import iplot # to make interactive visualizations
import seaborn as sns # data visualization library based on matplotlib
import scipy as sp # to solve scientific/mathematical problems
from scipy import stats # to perform statistical functions
import os # to call functions witin operating system
import datetime as dt # to work with dates as date objects
from matplotlib.colors import ListedColormap # to use a colormap
from pandas.plotting import scatter_matrix # to draw a matrix of scatter plots
from sklearn.cluster import KMeans # to perform k-means clustering
from sklearn.preprocessing import StandardScaler # to standardize features by removing the mean/scaling to unit variance
from lifetimes import BetaGeoFitter # to simulate customer purchases
from lifetimes.utils import summary_data_from_transaction_data # to calculate summary values for CLV
from lifetimes.plotting import plot_frequency_recency_matrix # to plot recency frequecy matrix heatmap 
from lifetimes.plotting import plot_period_transactions # to plot figure with period actual/predicted transactions.
from lifetimes.utils import calibration_and_holdout_data # to create summary of each customer over a calibration/holdout period
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases # to plot calibration purchases vs holdout
from lifetimes import GammaGammaFitter # to estimate average monetary value of customer transactions
from mlxtend.frequent_patterns import apriori # apriori function to extract frequent itemsets for association rule mining
from mlxtend.frequent_patterns import association_rules # function to generate association rules from frequent itemsets

import warnings


warnings.filterwarnings("ignore")


# In[3]:


salesdata = pd.read_csv('/Users/Nikhil/Desktop/DataScienceProjects/eCommerce/Online_Retail.csv', encoding="ISO-8859-1",dtype={'CustomerID': str,})


# # Data Overview

# In[4]:


# first 5 rows of dataset
salesdata.head()


# In[5]:


# last 5 rows of dataset
salesdata.tail()


# In[6]:


# count number of rows and columns in dataframe
salesdata.shape


# In[7]:


# dataset overview
salesdata.info()


# In[8]:


# data types in the dataframe
salesdata.dtypes


# In[9]:


# summary of statistics
salesdata.describe()


# In[10]:


# data by percentiles
salesdata.describe(percentiles = [.01, .05, .10, .25, .5, .75, .9, .95, .99])


# In[11]:


plt.style.use('seaborn')
sns.boxplot(salesdata.Quantity)


# In[12]:


plt.style.use('seaborn')
sns.distplot(salesdata['Quantity'].value_counts(), color = "b", kde = False)


# In[13]:


plt.style.use('seaborn')
sns.boxplot(salesdata.UnitPrice)


# In[14]:


plt.style.use('seaborn')
sns.distplot(salesdata['UnitPrice'].value_counts(), color="b", kde = False)


# If you look at the distribution for columns 'Quantity' and 'UnitPrice' in the box plot and distribution plot, you will notice that most of the data values are less than two digits (at least 50% are single-digit values for 'Quantity' and 'UnitPrice'). Both columns also consist of negative values. If you look at the minimum and maximum values for 'Quantity' you will notice that those values are significantly less than or exceed 99% of the data. The same can be said for 'UnitPrice', as the min and max value are considered to be outliers.

# In[15]:


# validate negative quantity values and unit prices
salesdata[salesdata['UnitPrice'] >= 8000].head(10)


# I decided to look into what the negative values in the dataset are a result of. It is clear that negative values for Quantity and high unit prices are linked with unusual values for 'Stockcode'. Such values will likely be removed at the data cleaning step.

# # Data Cleaning

# In[16]:


# check missing values
missing_values = salesdata.isna().sum().sort_values(ascending = False)
print(missing_values)

print('---------------------------')

# check percentage of missing values for each column
print(missing_values / len(salesdata))

print('---------------------------')

# view rows with missing values
missing_val_rows = salesdata[salesdata.isna().any(axis = 1)]
print(missing_val_rows.head())

print('---------------------------') 

# validate if there are any negative values in Quantity column
salesdata[salesdata['Quantity']<0].head()


# There are 136,534 missing values in the dataset for 'CustomerID' and 'Description'. Nearly 25% of the missing values are from the 'CustomerID' column. RFM Analysis, a customer segmentation technique that will be used further into the analysis, divides customers into different clusters. An effective approach would be to remove the missing values instead of imputation. Not removing such values could result in a more biased model, including the incorrect classification of data points. Missing values can also slow down the training time for each model. You will also notice in the output above that all of the rows have a negative value of quantity and InvoiceNo that begin with 'C', meaning that the order has been cancelled.

# In[17]:


# number of duplicates (all columns)
duplicate_all_rows = salesdata.duplicated().sum()
print("Number of Duplicate Rows in All Columns are:",(duplicate_all_rows))

# number of duplicates (select columns)
duplicate_subset_rows = salesdata.duplicated(subset = ['InvoiceNo', 'StockCode', 'InvoiceDate', 'CustomerID'], keep = 'first').sum()
print("Number of Duplicate Rows in Select Columns are:",(duplicate_subset_rows))


# There are 5,268 duplicate rows in the dataset with identical values based on all columns. However, when 'InvoiceNo', 'StockCode,' 'InvoiceDate,' and 'CustomerID' are subsetted, the number of duplicate rows in the dataset increases to 10,677. This does not mean the values in the other columns have matching values. As a result, 5,268 duplicate rows were removed as all columns were evaluated.

# In[18]:


# create function that converts all cancelled orders to 1 and everything else to 0 
def order_status(x):
    status = 0
    if x['InvoiceNo'][0] == 'C':
        status = 1
    return status

# use apply function to create a new column
salesdata['OrderStatus'] = salesdata.apply(order_status, axis = 1)


# In[19]:


# pie chart visualizing percentage of sales by day of week - non-uk customers
plt.style.use('seaborn')
explode = [0,0]
plt.figure(figsize = (10,7))
order_status_pie = salesdata['OrderStatus'].value_counts()
labels = ['PROCESSED', 'CANCELLED']
order_status_pie.plot(kind = 'pie', 
                 autopct = '%1.1f%%',
                 labels = labels, 
                 explode = explode, 
                 shadow = True)
plt.title('Order Status', weight = 'bold')
plt.axis('equal')
plt.ylabel('')
plt.show()


# The output above shows that there were 9,288 or 1.7% of orders that were cancelled by customers and 532,621 or 98.3% orders that were processed.

# In[20]:


# remove missing values
salesdata_filtered = salesdata.dropna()

# remove duplicates columns
salesdata_filtered = salesdata_filtered.drop_duplicates(keep = 'first')

# remove quantity with negative values
salesdata_filtered = salesdata_filtered[salesdata_filtered['Quantity']>0]

# remove cancelled orders
cancelled = salesdata_filtered[salesdata_filtered['InvoiceNo'].astype(str).str.contains('C')]

# check number of records remaining
salesdata_filtered.info()

print('----------------------------------------')

# check first five rows of dataframe
salesdata_filtered.head()


# In[21]:


# check missing values
print(salesdata_filtered.isna().sum().sort_values(ascending = False))

print('---------------------------')

# validate if there are any negative values in Quantity column

print(salesdata_filtered.Quantity.min())

print('---------------------------')

# validate if there are any negative values in UnitPrice column
print(salesdata_filtered.UnitPrice.min())


# In[22]:


# validate data cleaning for negative values
salesdata_filtered.describe()


# In[23]:


# print unusual stockcodes
spec_stockcodes = salesdata_filtered[salesdata_filtered['StockCode'].str.contains('^[a-zA-Z]+', regex = True)]['StockCode'].unique()
print(spec_stockcodes)


# In[24]:


# validate stockcode 'POST'
salesdata_filtered[salesdata_filtered['StockCode'] == 'POST'].head(3)


# In[25]:


# validate stockcode 'C2'
salesdata_filtered[salesdata_filtered['StockCode'] == 'C2'].head(3)


# In[26]:


# validate stockcode 'M'
salesdata_filtered[salesdata_filtered['StockCode'] == 'M'].head(3)


# In[27]:


# validate stockcode ' BANK CHARGES'
salesdata_filtered[salesdata_filtered['StockCode'] == 'BANK CHARGES'].head(3)


# In[28]:


# validate stockcode 'PADS'
salesdata_filtered[salesdata_filtered['StockCode'] == 'PADS'].head(3)


# In[29]:


# validate stockcode 'DOT'
salesdata_filtered[salesdata_filtered['StockCode'] == 'DOT'].head(3)


# At the Data Overview step, I discovered that there were values in the 'StockCode' column that are unusual. These values still remain in the data even after missing and negative values were removed.

# # Feature Engineering

# In[30]:


# change column names
salesdata_filtered.rename(index = str, columns = {'InvoiceNo': 'invoice_no',
                                                  'StockCode' : 'stock_code',
                                                  'Description' : 'description',
                                                  'Quantity' : 'quantity',
                                                  'InvoiceDate' : 'invoice_date',
                                                  'UnitPrice' : 'unit_price',
                                                  'CustomerID' : 'customer_id',
                                                  'Country' : 'country',
                                                  'OrderStatus' : 'order_status'}, inplace = True)

# re-format date/time column
salesdata_filtered.invoice_date = pd.to_datetime(salesdata_filtered.invoice_date, format = '%Y-%m-%d %H:%M')

# convert all lowercase characters in a string to uppercase characters using .str.upper()
salesdata_filtered['country'] = salesdata_filtered['country'].str.upper()

salesdata_filtered['stock_code'] = salesdata_filtered['stock_code'].str.upper()

# convert datatypes using .astype()
salesdata_filtered['invoice_no'] = salesdata_filtered['invoice_no'].str.upper().astype(str)

# remove all the leading and trailing spaces from string using str.strip() 
# remove all occurrences of character X from a string and replace with space using str.replace()
salesdata_filtered['description'] = salesdata_filtered['description'].str.replace('.','').str.upper().str.strip()

# remove any extra whitespace
salesdata_filtered['description'] = salesdata_filtered['description'].replace('\s+',' ', regex = True)

# convert to integer
salesdata_filtered['customer_id'] = pd.to_numeric(salesdata_filtered['customer_id'])

# create total sales column
total_sales_col = salesdata_filtered["quantity"] * salesdata_filtered["unit_price"]

salesdata_filtered.insert(loc = 6, column = "total_sales", value = total_sales_col)

# create year column
year_col = salesdata_filtered['invoice_date'].dt.year

salesdata_filtered.insert(loc = 9, column = 'year', value = year_col)

# create day of week and hour column
salesdata_filtered.insert(loc = 10, column = 'day_of_week', value = (salesdata_filtered.invoice_date.dt.dayofweek)+1)

salesdata_filtered.insert(loc = 11, column = 'hour', value = salesdata_filtered.invoice_date.dt.hour)

# create day of month column
day_of_month_col = salesdata_filtered['invoice_date'].dt.day

salesdata_filtered.insert(loc = 12, column = 'day_of_month', value = day_of_month_col)

# subset time period 2010-12-01 to 2011-11-30 (12 months) for filtered data 
salesdata_filtered = salesdata_filtered[~(salesdata_filtered['invoice_date'] > '2011-12-01')]

# group data by season
salesdata_filtered.loc[salesdata_filtered['invoice_date']<dt.date(2010,12,21),'season'] = 'AUTUMN'

salesdata_filtered.loc[((salesdata_filtered["invoice_date"]>=dt.date(2010,12,21)) & (salesdata_filtered["invoice_date"]<dt.date(2011,3,20))), 'season'] = "WINTER" 

salesdata_filtered.loc[((salesdata_filtered["invoice_date"]>=dt.date(2011,3,20)) & (salesdata_filtered["invoice_date"]<dt.date(2011,6,21))), 'season'] = "SPRING" 

salesdata_filtered.loc[((salesdata_filtered["invoice_date"]>=dt.date(2011,6,21)) & (salesdata_filtered["invoice_date"]<dt.date(2011,9,23))),'season'] = "SUMMER" 

salesdata_filtered.loc[((salesdata_filtered["invoice_date"]>=dt.date(2011,9,23)) & (salesdata_filtered["invoice_date"]<dt.date(2011,12,22))),'season'] = "AUTUMN"

# rearrange columns
salesdata_filtered = salesdata_filtered[["customer_id", "country", "invoice_no", "stock_code", "quantity", "unit_price", "total_sales", "description", "invoice_date", "year", "day_of_week", "hour", "day_of_month", "season", 'order_status']]

# remove special stockcodes
spec_stockcodes_values = ['POST', 'C2', 'M', 'BANK CHARGES', 'PADS', 'DOT']
salesdata_filtered = salesdata_filtered[~salesdata_filtered['stock_code'].isin(spec_stockcodes_values)]

# create dataframe with only UK customers
uk_data = salesdata_filtered[salesdata_filtered.country == 'UNITED KINGDOM']

# create dataframe without UK customers
data_not_uk = salesdata_filtered[salesdata_filtered.country != 'UNITED KINGDOM']

# check number of records in UK and non-UK customer data 
print(salesdata_filtered.info())
print('---------------------------')
print(uk_data.info())
print('---------------------------')
print(data_not_uk.info())


# # Exploratory Data Analysis - EDA

# In[31]:


# number of transactions by UK customers
print('Number of transactions by UK Customers:', 
      uk_data['invoice_no'].nunique())

# number of products bought by UK customers
print('Number of products purchased by UK Customers:', 
      uk_data['stock_code'].nunique())

# number of customers in UK
print('Number of customers in UK:', 
      uk_data['customer_id'].nunique())

# total sales - UK
print('Total Sales Earned from UK Customers in 2011:', 
      uk_data['total_sales'].sum())

# total sales - Non-UK
print('Total Sales Earned from Non-UK Customers in 2011:', 
      data_not_uk['total_sales'].sum())


# In[32]:


# bar plot visualizing relationship between total sales and orders processed by country
plt.style.use('ggplot')
country_orders = salesdata_filtered.groupby('country')['invoice_no'].count().sort_values(ascending = False)
del country_orders['UNITED KINGDOM']
country_sales = salesdata_filtered.groupby('country')['total_sales'].sum().sort_values(ascending = False)
del country_sales['UNITED KINGDOM']
fig,axs = plt.subplots(nrows = 2, ncols = 1, figsize = (12,14))
fig.suptitle('Total Sales & Orders by Country')
sns.barplot(x = country_orders.values, y = country_orders.index, palette = ("cubehelix"), ax = axs[0]).set(xlabel = 'Number of Orders by Country')
sns.barplot(x = country_sales.values, y = country_sales.index, palette = ("cubehelix"), ax = axs[1]).set(xlabel = 'Total Sales by Country')
plt.show()


# In[33]:


# bar plot visualizing relationship between total sales and number of products sold - UK customers
plt.style.use('seaborn')
uk_data['month'] = uk_data['invoice_date'].dt.month_name() 
monthly_sales_quant_uk = uk_data.groupby(['month', 'year'])['total_sales', 'quantity'].sum().sort_values(by = ['total_sales', 'quantity'], ascending = False)
monthly_sales_quant_uk.plot(kind = 'bar', colormap='PRGn', alpha=0.85, legend = True, figsize = (10,5))
plt.xlabel('Month, Year', weight = 'bold')
plt.ylabel('Count', weight = 'bold')
plt.title('Total Sales and Number of Products Sold by Month, Year - UK Customers', weight = 'bold')
plt.show()

# bar plot visualizing relationship between total sales and number of products sold - non-UK customers
plt.style.use('seaborn')
data_not_uk['month'] = data_not_uk['invoice_date'].dt.month_name() 
monthly_sales_quant_not_uk = data_not_uk.groupby(['month', 'year'])['total_sales', 'quantity'].sum().sort_values(by=['total_sales', 'quantity'], ascending = False)
monthly_sales_quant_not_uk.plot(kind = 'bar', colormap='PRGn', alpha=0.85, legend = True, figsize = (10,5))
plt.xlabel('Month, Year', weight = 'bold')
plt.ylabel('Count', weight = 'bold')
plt.title('Total Sales and Number of Products Sold by Month, Year - Non-UK Customers', weight = 'bold')

plt.show()


# In[34]:


# bar plot visualizing relationship between total sales and number of products sold by hour - UK customers
plt.style.use('seaborn')
hour_sales_quant_uk = uk_data.groupby('hour')['total_sales', 'quantity'].sum().sort_values(by=['total_sales', 'quantity'], ascending = True)[5:15]
hour_sales_quant_uk.plot(kind = 'barh', color=['gray', 'black'], legend=True, figsize = (10,5))
plt.xlabel('Total Sales', weight = 'bold')
plt.ylabel('',weight = 'bold')
plt.title('Total Sales and Number of Products Sold by Hour - UK Customers',weight='bold')
plt.show()

# bar plot visualizing relationship between total sales and number of products sold by hour - non-UK customers
plt.style.use('seaborn')
hour_sales_quant_not_uk = data_not_uk.groupby('hour')['total_sales', 'quantity'].sum().sort_values(by = ['total_sales', 'quantity'], ascending = True)[5:15]
hour_sales_quant_not_uk.plot(kind = 'barh', color=['gray', 'black'], legend = True, figsize = (10,5))
plt.xlabel('Total Sales',weight = 'bold')
plt.ylabel('', weight = 'bold')
plt.title('Total Sales and Number of Products Sold by Hour - Non-UK Customers', weight = 'bold')
plt.show()


# In[35]:


# scatter plot visualizing relationship between number of products sold and unit price by season - UK customers
sales_scatter_uk = uk_data[(uk_data["quantity"]<=1500) & (uk_data["total_sales"]<=2000)]
plt.figure(figsize = (10,7))
sns.set_style("darkgrid")
sns.scatterplot(x = "quantity", y = "total_sales", hue = "season",
                data = sales_scatter_uk, 
                palette = ['green','blue','red', 'purple'], legend = 'full', alpha = 0.75, edgecolor = 'black', linewidth = 1,)
plt.xlabel('Quantity', weight = 'bold')
plt.ylabel('Total Sales', weight = 'bold')
plt.title('Number of Products Sold by Total Sales - UK Customers', weight = 'bold')
plt.xticks()
plt.show()


# In[36]:


# scatter plot visualizing relationship between number of products sold and total sales by season - non-UK customers
sales_scatter_not_uk = data_not_uk[(data_not_uk["quantity"]<=1500) & (data_not_uk["total_sales"]<=2000)]
plt.figure(figsize = (10,7))
sns.set_style("darkgrid")
sns.scatterplot(x = "quantity", y = "total_sales", hue = "season",
                data = sales_scatter_not_uk, 
                palette = ['green','blue','red', 'purple'], legend = 'full', alpha = 0.75, edgecolor = 'black', linewidth = 1,)
plt.xlabel('Quantity', weight = 'bold')
plt.ylabel('Total Sales', weight = 'bold')
plt.title('Number of Products Sold by Total Sales - Non-UK Customers', weight = 'bold')
plt.xticks()
plt.show()


# In[37]:


# bar plot visualizing relationship between total sales and season - non-UK and UK customers
season_sales_not_uk = data_not_uk.groupby('season')['total_sales'].sum().sort_values(ascending = False)
season_sales_uk = uk_data.groupby('season')['total_sales'].sum().sort_values(ascending = False)
plt.style.use('seaborn')
fig, axs = plt.subplots(nrows = 2, ncols = 1)
plt.subplots_adjust(hspace = 0.5)
plt.suptitle('Total Sales by Season')
sns.barplot(x = season_sales_not_uk.values, y = season_sales_not_uk.index, ax = axs[0]).set(xlabel = 'Total Sales Non-UK Customers')
sns.barplot(x = season_sales_uk.values, y = season_sales_uk.index, ax = axs[1]).set(xlabel = 'Total Sales - UK Customers')
plt.show()


# In[38]:


# scatter plot visualizing relationship between number of products sold and unit price by season - non-UK customers
quantity_scatter_not_uk = data_not_uk[(data_not_uk["quantity"]<=500) & (data_not_uk["unit_price"]<=50)]
plt.figure(figsize = (10,7))
sns.set_style("darkgrid")
sns.scatterplot(x = "quantity", y = "unit_price", hue = "season",
                data = quantity_scatter_not_uk, 
                palette = ['green', 'blue', 'red', 'purple'], legend = 'full', alpha = 0.75, edgecolor = 'black', linewidth = 1,)
plt.xlabel('Quantity', weight = 'bold')
plt.ylabel('Unit Price', weight = 'bold')
plt.title('Number of Products Sold by Unit Price - Non-UK Customers', weight = 'bold')
plt.xticks()
plt.show()

# scatter plot visualizing relationship between number of products sold and unit price by season - UK customers
quantity_scatter_uk = uk_data[(uk_data["quantity"]<=500) & (uk_data["unit_price"]<=50)]
plt.figure(figsize = (10,7))
sns.set_style("darkgrid")
sns.scatterplot(x = "quantity", y = "unit_price", hue = "season",
                data = quantity_scatter_uk, 
                palette = ['green','blue','red', 'purple'], legend='full', alpha = 0.75, edgecolor = 'black', linewidth=1,)
plt.xlabel('Quantity', weight = 'bold')
plt.ylabel('Unit Price', weight = 'bold')
plt.title('Number of Products Sold by Unit Price - UK Customers', weight = 'bold')
plt.xticks()
plt.show()


# In[39]:


# bar plot visualizing relationship between total sales by day of week - non-UK customers
plt.style.use('seaborn')
fig, ax1 = plt.subplots(figsize=(10,5))
daily_sales_not_uk = data_not_uk.groupby('day_of_week')['total_sales'].sum().reset_index()
ax1.set_title('Total Sales by Day of Week - Non-UK Customers')
sns.barplot(x = 'day_of_week', y = 'total_sales', data = daily_sales_not_uk)
ax1.set_xticklabels(('Monday', 'Tuesday', "Wednesday", "Thursday", "Friday", "Sunday"))
plt.show()

# pie chart visualizing percentage of revenue by day of week - UK customers
plt.style.use('seaborn')
explode = [0,0,0,0,0,0.0] 
plt.subplots(figsize = (10,7))
daily_sales_uk = uk_data.groupby('day_of_week')['total_sales'].sum()
labels = ['Monday', 'Tuesday', "Wednesday", "Thursday", "Friday", "Sunday"]
daily_sales_uk.plot(kind = 'pie', 
                    autopct = '%1.1f%%', 
                    labels = labels, 
                    explode = explode, 
                    shadow = True)
plt.title('Total Sales by Day of Week - UK Customers', weight = 'bold')
plt.axis('equal')
plt.ylabel('')
plt.show() 


# In[40]:


# heatmap visualizing relationship between total sales by month and day of week - Non-UK customers
plt.style.use('seaborn')
sales_month_day_heatmap_not_uk = data_not_uk.pivot_table(index = 'month', columns = 'day_of_week', values = 'total_sales', aggfunc = 'sum')
sns.heatmap(sales_month_day_heatmap_not_uk, cmap = 'OrRd')
plt.xlabel('Day of Week', weight = 'bold')
plt.ylabel('Month', weight = 'bold')
plt.title('Total Sales by Month and Day of Week - Non-UK Customers', weight = 'bold')
plt.xticks()
plt.show()

# heatmap visualizing relationship between total sales by month and day of week - UK customers
plt.style.use('seaborn')
sales_month_day_heatmap_uk = uk_data.pivot_table(index = 'month', columns = 'day_of_week', values = 'total_sales', aggfunc = 'sum')
sns.heatmap(sales_month_day_heatmap_uk, cmap = 'OrRd')
plt.xlabel('Day of Week', weight = 'bold')
plt.ylabel('Month', weight = 'bold')
plt.title('Total Sales by Month and Day of Week - UK Customers', weight = 'bold')
plt.xticks()
plt.show()

# heatmap visualizing relationship between total sales by month and day of month - non-UK customers
plt.style.use('seaborn')
uk_data['month'] = uk_data['invoice_date'].dt.month_name() 
sales_day_of_month_heatmap_not_uk = data_not_uk.groupby(['day_of_month'])['total_sales', 'quantity'].sum().sort_values(by = ['day_of_month'])
sales_day_of_month_heatmap_not_uk.plot(kind = 'bar', colormap='PRGn', alpha=0.85, legend = True, figsize = (10,5))
plt.xlabel('Day of Day of Month', weight = 'bold')
plt.ylabel('Count',weight = 'bold')
plt.title('Total Sales by Day of Month - Non-UK Customers', weight = 'bold')
plt.xticks()
plt.show()

# heatmap visualizing relationship between total sales by month and day of month - UK customers
plt.style.use('seaborn')
uk_data['month'] = uk_data['invoice_date'].dt.month_name() 
sales_day_of_month_heatmap_uk = uk_data.groupby(['day_of_month'])['total_sales', 'quantity'].sum().sort_values(by = ['day_of_month'])
sales_day_of_month_heatmap_uk.plot(kind = 'bar', colormap='PRGn', alpha=0.85, legend = True, figsize = (10,5))
plt.xlabel('Day of Day of Month', weight = 'bold')
plt.ylabel('Count',weight = 'bold')
plt.title('Total Sales by Day of Month - UK Customers', weight = 'bold')
plt.xticks()
plt.show()


# In[41]:


# bar plot visualizing products considered as "best sellers" by total sales - non-UK and UK customers
product_best_sellers_not_uk = data_not_uk.groupby('description')['total_sales'].sum().sort_values(ascending = False).head(30)
product_best_sellers_uk = uk_data.groupby('description')['total_sales'].sum().sort_values(ascending = False).head(30)
plt.style.use('ggplot')
fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (12,14))
fig.suptitle('Best Sellers by Total Sales')
sns.barplot(x = product_best_sellers_not_uk.values, y = product_best_sellers_not_uk.index, palette = ("Blues_r"), ax = axs[0]).set(xlabel = 'Total Sales Non-UK Customers')
sns.barplot(x = product_best_sellers_uk.values, y = product_best_sellers_uk.index, palette = ("Blues_r"), ax = axs[1]).set(xlabel = 'Total Sales - UK Customers')
plt.show()


# In[42]:


# average number of orders per non-uk customer
print('average number of orders per non-uk customer:')
avg_ord_customers_not_uk = data_not_uk .groupby('customer_id')['invoice_no'].nunique()
print(avg_ord_customers_not_uk.describe())
print('---------------------------')

# average number of orders per uk customer
print('average number of orders per uk customer:')
avg_ord_customers_uk = uk_data.groupby('customer_id')['invoice_no'].nunique()
print(avg_ord_customers_uk.describe())
print('---------------------------')

# average number of unqiue items per order and per non-uk customer
print('average number of unqiue items per order and per non-uk customer:')
groupby_invoice_not_uk = data_not_uk.groupby('invoice_no')['stock_code'].nunique()
print(groupby_invoice_not_uk.describe())
print('---------------------------')

# average number of unqiue items per order and per uk customer
print('average number of unqiue items per order and per uk customer:')
groupby_invoice_uk = uk_data.groupby('invoice_no')['stock_code'].nunique()
print(groupby_invoice_uk.describe())
print('---------------------------')

# number of unique products per non-uk customer
print('number of unique products per non-uk customer:')
groupby_ID_not_uk = data_not_uk.groupby('customer_id')['stock_code'].nunique()
print(groupby_ID_not_uk.describe())
print('---------------------------')

# number of unique products per uk customer
print('number of unique products per uk customer:')
groupby_ID_uk = uk_data.groupby('customer_id')['stock_code'].nunique()
print(groupby_ID_uk.describe())


# - UK, Netherlands, and Eire recorded the highest sales volume.
# - UK, Germany, and France processed the highest number of shopping orders.
# - Monthly sales analysis shows that January, November, and June recorded the highest revenue between December 2010 and November 2011 in the UK. 
# - Most customers shopped online during the lunch hour and in the early hours of the morning in the UK.
# - Autumn was the biggest season for sales. As the holiday season ended sales took a nosedive in the UK.
# - The beginning of the month and the middle of the month recorded the highest sales volume.
# - Weekly sales analysis shows that Thursday recorded the highest revenue in the UK.
# - The average number of orders per customer in the UK and outside the UK is 4.
# - The average number of unique products purchased in the UK per order is 20.
# - The average number of unique products per customer is 59 in the UK.

# # Data Modeling - RFM Analysis
# 
# RFM Analysis is used to better understand customer behaviour, map customer journeys, make personalized product recommendations, and deliver excellent customer experience. As part of the RFM Analysis 'customer_id' will be used to identify individual customers; 'invoice_date' will be used to calculate how recent a transaction was completed by a customer; 'invoice_no' will be used to calculate the number of transactions that were completed by a customer; 'quantity' will be used to calculate how many products were purchased by a customer, and 'unit_price' will be used to calculate the total amount of money spent by a customer through 2010-12-01 to 2011-11-30.

# In[49]:


# check max and min date of dataset
uk_data['invoice_date'].min(), uk_data['invoice_date'].max()

# set recent date to December 1, 2011
recent_date = dt.datetime(2011,12,1)

# create rfm scores for each customer
rfm_scores = uk_data.groupby('customer_id').agg({'invoice_date': lambda x: (recent_date - x.max()).days,
                                                 'invoice_no': lambda x: len(x.unique()),
                                                 'total_sales': lambda x: x.sum()})

# convert invoice_date into integer
rfm_scores['invoice_date'] = rfm_scores['invoice_date'].astype(int)

# rename column names to recency, frequency and monetary
rfm_scores.rename(columns = {'invoice_date': 'recency', 
                             'invoice_no': 'frequency', 
                             'total_sales': 'monetary_value'}, inplace=True)

# first 5 rows of dataset
rfm_scores.reset_index().head(10)


# The original dataset has transactions between December 1, 2010, and December 9, 2011. For our dataset I will be condensing our transaction history to December 1, 2010, to November 30, 2011). This one-year subset is important when calculating the recency value of a transaction completed by each customer.

# In[50]:


# validate frequency
uk_data[uk_data['customer_id'] == 12820.0]


# The output above shows that there are duplicate rows with the identical Customer ID and Invoice Number, which have to be removed because it is considered one purchase.

# In[51]:


# remove duplicates for invoice_no and customer_id columns
uk_data.drop_duplicates(subset = ['invoice_no', 'customer_id'], keep = "first", inplace = True)


# In[52]:


# validate frequency
uk_data[uk_data['customer_id'] == 12820.0]


# After removing duplicate rows with the identical Customer ID and Invoice Number, there are now duplicate rows with the same Customer ID but with a different Invoice Number.

# In[53]:


# descriptive statistics (recency)
print('average number of days between between present date and date of last purchase:')
print(rfm_scores.recency.describe())
print('---------------------------')

# descriptive statistics (frequency)
print('average number of transactions that were made:')
print(rfm_scores.frequency.describe())
print('---------------------------')

# descriptive dtatistics (monetary value)
print('average money spent:')
print(rfm_scores.monetary_value.describe())


# - The average number of days between the present date and date of the last purchase for each customer is 90 days.
# - The average number of transactions made by each customer is 4.
# - The average number of money spent by each customer is 1,752.93.

# In[54]:


# split into four segments using quantiles
quantiles = rfm_scores.quantile(q = [0.25,0.50,0.75])
print(quantiles)
print('---------------------------')
# convert quantiles to dictionary
quantiles = quantiles.to_dict()
print(quantiles)


# In[55]:


# create function for recency where 1 is assigned to the lowest level of recency
def rclass_score(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4

# create function for frequency and monetary where 1 is assigned to the highest value
def fmclass_score(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1


# For the two functions above, x represents every row for the column; d refers to the quantiles and prefers to the column name (recency, frequency, and monetary value).

# In[56]:


# apply functions to dataframe to calculate r, f, m segment values
rfm_scores['r_quartile'] = rfm_scores['recency'].apply(rclass_score, args = ('recency', quantiles,))
rfm_scores['f_quartile'] = rfm_scores['frequency'].apply(fmclass_score, args = ('frequency', quantiles,))
rfm_scores['m_quartile'] = rfm_scores['monetary_value'].apply(fmclass_score, args = ('monetary_value', quantiles,))
rfm_scores.reset_index().head(10)


# In[57]:


# create column by concatenating each scores into one value using .map()
rfm_scores['rfm_class'] = rfm_scores.r_quartile.map(str) + rfm_scores.f_quartile.map(str) + rfm_scores.m_quartile.map(str)


# In[58]:


# create column of sum values for r_quartile', f_quartile, and m_quartile
rfm_scores['rfm_score'] = rfm_scores[['r_quartile', 'f_quartile', 'm_quartile']].sum(axis = 1)
rfm_scores.reset_index().head(10)


# If a customer scored a low RFM score then their engagement and loyalty are considered strong. For example, customer 12346.0 has an RFM score of 9. This is due to this specific customer making their one and only transaction 316 days ago. Another example would be customer 12748 who has an RFM score of 3 which is a product of having a low recency value of 1 and high-frequency value. 

# In[59]:


# create four loyalty levels
loyalty_level = ['CHAMPIONS', 'PROMISING', 'AT_RISK', 'CHURNED'] 

# create score cuts based on pandas .qcut() method
score_cuts = pd.qcut(rfm_scores.rfm_score, q = 4, labels = loyalty_level)

# create new loyalty level column 
rfm_scores['loyalty_level'] = score_cuts.values
rfm_scores.reset_index().head(10)


# # Model Validation

# ### Champions
# Customers classified as 'champions' are those who scored between 3 and 5. They purchased recently, often, and spent the most.

# In[60]:


# validate data for customers with loyalty level: champions
validate_champions = rfm_scores[rfm_scores['loyalty_level'] == 'CHAMPIONS'].sort_values('monetary_value', ascending = False)
validate_champions.reset_index().head(10)


# ### Promising
# Customers classified as 'promising' are those who scored between 6 and 7. They purchased recently but have not spent much.

# In[61]:


# validate data for customers with loyalty level: promising
validate_promising = rfm_scores[rfm_scores['loyalty_level'] == 'PROMISING'].sort_values('monetary_value', ascending = False).reset_index()
validate_promising.reset_index().head(25)


# ### At-Risk 
# Customers classified as 'at-risk' are those who scored between 9 and 10. They spent a lot of money, purchased often, but not recently.

# In[62]:


# validate data for customers with loyalty level: at-risk
validate_at_risk = rfm_scores[rfm_scores['loyalty_level'] == 'AT_RISK'].sort_values('monetary_value', ascending = False).reset_index()
validate_at_risk.reset_index().head(25)


# ### Churned 
# Customers classified as 'churned' are those who scored between 11 and 12. They are low in recency, frequency and monetary.

# In[63]:


# validate data for customers with loyalty level: churned
validate_churned = rfm_scores[rfm_scores['loyalty_level'] == 'CHURNED'].sort_values('monetary_value', ascending = False).reset_index()
validate_churned.reset_index().head(25)


# In[64]:


# remove columns to visualize relationship between recency, frequency, and monetary values
rfm_corr = rfm_scores.drop(['r_quartile', 'f_quartile', 'm_quartile', 'rfm_class', 'rfm_score', 'loyalty_level'], axis = 1)

# heatmap visualization for rfm correlation
plt.style.use('ggplot')
plt.figure(figsize = (10,7))
sns.heatmap(rfm_corr.corr(), annot = True, cmap = 'coolwarm', linecolor = 'white', linewidths = .5)
plt.title('RFM Correlation', weight = 'bold')
plt.xticks()
plt.show()


# - There is a very weak correlation between recency and frequency.
# - There is a weak correlation between recency and monetary. 
# - There is a moderate correlation between frequency and monetary.

# In[65]:


# scatter matrix visualizing relationship between recency, frequency and monetary values (pre-normalization)
sns.set_style("whitegrid")
pd.plotting.scatter_matrix(rfm_corr, alpha = .8, diagonal = 'kde', c = 'blue', s = 10, figsize = (14,8))


# ## Preprocessing Data
# 
# From the above and during EDA I noticed that it is a skewed distribution for recency, frequency, and monetary values. In order to reduce the skewness in the data, I need to apply log transformation using the np.log() method to bring the data to near-normal distribution. Then StandardScaler() will be applied to the features by removing the mean and scaling to unit variance. However, I need to deal with negative or zero values prior to applying the log transformation by creating a function that will return 1 if x is less than or equal to zero to avoid getting any errors when applying log transformation and to enhance the overall performance of k-means clustering algorithm.

# In[66]:


# create function to handle with negative and zero values
def negative_zero(x):
    if x <= 0:
        return 1
    else:
        return x
    
# apply function to recency and monetary_value columns 
rfm_scores['recency'] = [negative_zero(x) for x in rfm_scores.recency]
rfm_scores['monetary_value'] = [negative_zero(x) for x in rfm_scores.monetary_value]

# perform log transformation to bring data into normal or near normal distribution
normalized_data = rfm_scores[['recency', 'frequency', 'monetary_value']].apply(np.log, axis = 1)
print(normalized_data.head())

print('------------------------------------------------')

# create standardization scaling object
scaler = StandardScaler()

# fit standardization parameters and scale the data
scaled_data = scaler.fit_transform(normalized_data)
print(scaled_data)

# transform it back to dataframe
scaled_data = pd.DataFrame(scaled_data,columns = normalized_data.columns, index = rfm_scores.index)
scaled_data.head(10)


# In[67]:


# scatter matrix visualizing relationship between recency, frequency and monetary values (post-normalization)
sns.set_style("whitegrid")
pd.plotting.scatter_matrix(scaled_data, alpha = .8, diagonal = 'kde', c = 'blue', s = 10, figsize = (14,8))
plt.show()


# In[68]:


# create new dataframe to discover customer clusters 
rfm_scores_loyalty_cluster = rfm_scores.drop(['r_quartile', 'f_quartile', 'm_quartile', 'rfm_class', 'rfm_score'], axis=1)

# relationship between recency, frequency and monetary values.
sns.set_style("white")
rfm_scores_loyalty_cluster_plot = rfm_scores_loyalty_cluster[(rfm_scores_loyalty_cluster["recency"]<=300) & (rfm_scores_loyalty_cluster["frequency"]<=100) & (rfm_scores_loyalty_cluster["monetary_value"]<=5000)]
sns.pairplot(rfm_scores_loyalty_cluster_plot, hue = "loyalty_level", plot_kws = {'s': 20, 'edgecolor' : 'black'}, palette = "Set1")


# In[69]:


# calculate average values for each loyalty level
rfm_level_agg = rfm_scores.groupby(['loyalty_level'], as_index = False).mean().groupby('loyalty_level')['recency', 'frequency', 'monetary_value'].mean().round(0)
    
# print the aggregated dataset
rfm_level_agg.head()

# add count column into dataset
count = rfm_scores['loyalty_level'].value_counts()
rfm_level_agg.insert(loc = 3, column = "count", value = count)
rfm_level_agg.reset_index().head()


# In[70]:


# pie chart visualizing customer loyalty
plt.style.use('seaborn')
explode = [0,0,0,0] 
plt.subplots(figsize = (10,7))
loyalty_level_pie = rfm_level_agg.groupby('loyalty_level')['count'].sum()
labels = ['CHAMPIONS', 'PROMISING', "AT_RISK", "CHURNED"]
loyalty_level_pie.plot(kind = 'pie', 
                       autopct = '%1.1f%%', 
                       labels = labels, 
                       explode = explode,
                       shadow = True)
plt.title('Percentage of Loyalty Level', weight = 'bold')
plt.axis('equal')
plt.ylabel('')
plt.show() 


# - Customers classified as 'champions' have average recency of 19, the average frequency of 10, and the average monetary value of 4530. 
# - Customers classified as 'promising' have average recency of 59, the average frequency of 3, and the average monetary value of 1036. 
# - Customers classified as 'at-risk' have average recency of 95, the average frequency of 1, and average monetary value of 514. 
# - Customers classified as 'churned' have average recency of 220, the average frequency of 1, and average monetary value of 219. 

# ## K-Means Clustering
# 
# K-means clustering, an unsupervised learning algorithm, is a technique that is used to predict groupings from within an unlabeled dataset. 
# 
# First, a number of clusters 'k' is selected. 
# 
# 'K' distinct data points are randomly selected considered to be the centroid (centre point) of each cluster. 
# 
# The distance between each data point and the selected 'k' clusters is measured.
# 
# Each data point is assigned to the cluster for which the centroid is the closest or nearest mean: that with the least squared Euclidean distance. This step is repeated until the same points are assigned to each cluster consecutively.

# In[71]:


sse = {} # fit k-means and calculate SSE for each 'k'

for k in range(1,10): # loop will fit the k-means algorithm
    kmeans = KMeans(n_clusters = k, init = 'k-means++', n_init = 10, max_iter = 300) # number of clusters to create and the number of centroids to generate
    kmeans.fit(scaled_data) # compute k-means clustering on pre-processed data
    sse[k] = kmeans.inertia_ # sum of squared distances to closest cluster center


# In[72]:


# line graph to visualize elbow method to determine optimal value of k 
plt.style.use('ggplot')
plt.figure(figsize = (10,7))
plt.plot(list(sse.keys()), list(sse.values()))
plt.grid(True)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Square Distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# The Elbow Criterion method was used to select the optimal 'k' value. The idea behind the technique is to calculate the sum of squared errors (SSE) for each value of 'k'. From the output below you will notice from the line graph that as 'k' increases, SSE decreases. From the graph, I can see that 3 is the optimal number of clusters. 

# In[73]:


# build k-means model
KMean_clust = KMeans(n_clusters= 3, n_init=10, init= 'k-means++', max_iter= 300)
KMean_clust.fit(scaled_data)

# find the clusters for each data point given in the dataset
rfm_scores['cluster'] = KMean_clust.labels_ # extract cluster labels from labels_ attribute
rfm_scores.reset_index().head(10)


# In[74]:


rfm_scores_cluster = rfm_scores[(rfm_scores["recency"]<=300) & (rfm_scores["frequency"]<=100)]
sns.set_style("darkgrid")
plt.figure(figsize = (10,7))
sns.scatterplot(x = "recency", y = "frequency", hue = "cluster",
                data = rfm_scores_cluster, 
                palette = ['green','blue','red'], legend = 'full', alpha = 0.75, edgecolor = 'black', linewidth=1,)
plt.xlabel('Recency', weight = 'bold')
plt.ylabel('Frequency', weight = 'bold')
plt.title('Relationship Between Recency and Frequency Clusters', weight = 'bold')
plt.xticks()
plt.show()

# scatter plot frequency vs recency
rfm_scores_cluster = rfm_scores[(rfm_scores["frequency"]<=100) & (rfm_scores["monetary_value"]<=10000)]
sns.set_style("darkgrid")
plt.figure(figsize = (10,7))
sns.scatterplot(x="frequency", y="monetary_value", hue="cluster",
                data=rfm_scores_cluster, 
                palette=['green','blue','red'], legend='full', alpha = 0.75, edgecolor = 'black', linewidth=1,)
plt.xlabel('Frequency', weight = 'bold')
plt.ylabel('Monetary Value', weight = 'bold')
plt.title('Relationship Between Frequency and Monetary Value Clusters', weight = 'bold')
plt.xticks()
plt.show()

# scatter plot frequency vs recency
rfm_scores_cluster = rfm_scores[(rfm_scores["recency"]<300) & (rfm_scores["monetary_value"]<=10000)]
sns.set_style("darkgrid")
plt.figure(figsize = (10,7))
sns.scatterplot(x="recency", y="monetary_value", hue="cluster",
                data=rfm_scores_cluster, 
                palette=['green','blue','red'], legend='full', alpha = 0.75, edgecolor = 'black', linewidth=1,)
plt.xlabel('Recency', weight = 'bold')
plt.ylabel('Monetary Value', weight = 'bold')
plt.title('Relationship Between Recency and Monetary Value Clusters', weight = 'bold')
plt.xticks()
plt.show()


# From the output above you will notice a group of customers (green) who purchased recently, did not buy often. But when they bought recently and often their spending increases. 

# In[75]:


# calculate average values for each rfm
rfm_agg = rfm_scores.groupby(['rfm_score'], as_index = False).mean().groupby('rfm_score')['recency','frequency','monetary_value'].mean().round(0)
    
# print the aggregated dataset
rfm_level_agg.head()

# add count column into dataset
count = rfm_scores['rfm_score'].value_counts()
rfm_agg.insert(loc = 3, column = "count", value = count)
rfm_agg.reset_index().head()


# - The average number of days between the present date and date of last purchase for the customer(s) that achieved an RFM score of 3 is 6.
# - The average number of purchase frequency for the customer(s) that achieved an RFM score of 3 is 15.
# - Average total revenue generated for the customer(s) that achieved an RFM score of 3 is 7717.0

# ## Data Modeling - Customer Lifetime Value

# Customer Lifetime Value (CLTV) predicts the net dollar value that a company can attribute to the future business relationship with each customer. Through CLTV a company can determine how much revenue they can expect each customer could generate. 
# 
# From our previous analysis, I know that there are 3883 unique Customer IDs in the UK (the customer base that will be used to build CLV). I also know that the last order date in the data set is November 30, 2011.
# 
# 'Frequency represents the number of days a customer made a transaction on. 
# 
# 'Recency' represents the duration between a customer's first transaction and their most recent transaction. For example, if a customer had only one transaction, their recency value is equal to zero. 
# 
# 'T' represents the duration between a customer’s first transaction and the end of the period being evaluated.
# 
# 'Monetary_value' represents the average value of a given customer's purchases. 

# In[76]:


clv_data = summary_data_from_transaction_data(uk_data,'customer_id', 'invoice_date', 
                                              monetary_value_col = 'total_sales', 
                                              observation_period_end = '2011-11-30')
print(clv_data.reset_index().head())

print('---------------------------')

print(clv_data['frequency'].describe())

print('---------------------------')

print(clv_data['recency'].describe())

print('---------------------------')

print(clv_data['monetary_value'].describe())


# Since CustomerID 12346.0 made a purchase only once,  the value for frequency and recency will be 0 because this customer is not considered someone who buys often or a "repeat customer." Customer's made an average of 2.70 of repeat purchases. 

# ### Frequency-Recency Analysis: Beta Geometric / Negative Binomial distribution Model (BG/NBD)

# In[77]:


# fit bg model
bgf_mod = BetaGeoFitter(penalizer_coef = 0.1)
bgf_mod.fit(clv_data['frequency'], clv_data['recency'], clv_data['T'])
print(bgf_mod)

print('-------------------------------------')

plt.style.use('ggplot')
sns.set(rc={'image.cmap':'rainbow'})

# visualizing frequency/recency matrix
plt.figure(figsize = (10,5))
plot_frequency_recency_matrix(bgf_mod)


# The frequency-recency matrix helps us calculate the number of transactions I can expect a customer will make in the time period (day, week, or month). If a customer purchased 100 times and their most recent purchase came 350 days ago, then this customer is considered your 'champion' (bottom-right) or someone who more likely to return as a customer. If a customer purchased 20 times and their most recent purchase came 200 or 150 days ago, then this customer is someone who purchased recently but has not done so recently. Customers who are at risk of churning or they purchased in large volume but haven’t done so recently are usually located at the top-right corner of the matrix.

# ### Customer Rankings

# In[78]:


# predict future transaction in next 10 days
t = 10

clv_data['pred_txn'] = round(bgf_mod.conditional_expected_number_of_purchases_up_to_time(t, clv_data['frequency'], clv_data['recency'], clv_data['T']),2)

clv_data.sort_values(by = 'pred_txn', ascending = False).head(10).reset_index()


# When we rank customers from highest expected purchases in the next 10 days, we see that Customer 17841.0 made 107 purchases within 363 days (duration between a customer's first transaction and their most recent transaction) and is expected to make nearly three more purchases within the next 10 days.

# ### Cross-Validation

# In[79]:


summary_cal_holdout = calibration_and_holdout_data(uk_data, 'customer_id', 'invoice_date', 
                                                   calibration_period_end = '2011-05-30',
                                                   observation_period_end = '2011-11-30' ) 
summary_cal_holdout.head().reset_index()                                   


# To test the model we must divide the data set into balanced proportions such as a sample (calibration) and validation (holdout) set. We will fit the model on the calibration set (the beginning to 2011-05-30) to build the model. Meanwhile, the holdout set (2011-05-31 to 2011-11-30) will be used to validate the model on data not yet seen. The main idea is to never use the test set to evaluate a model’s performance using the ‘actual’ data.

# In[80]:


plt.style.use('ggplot')
bgf_mod.fit(summary_cal_holdout['frequency_cal'], 
            summary_cal_holdout['recency_cal'], 
            summary_cal_holdout['T_cal'])

plot_calibration_purchases_vs_holdout_purchases(bgf_mod, summary_cal_holdout, figsize = (10,7))


# From the output above you see customers from sample data plot by their repeat transactions rate (x-axis) and average over their repeat transactions in the validation set (y-axis). The model is able to linearly predict the customer's behaviour using the sample of the data. However, the model results in accurate predictions but underestimates at 4 purchases and overestimates at 5 purchases, but underestimates at 4 and 6 purchases. 

# ## Customer Lifetime Value: Gamma-Gamma Model

# In[81]:


# subset customers who had at least one repeat purchase
print(sum(clv_data['frequency'] == 0)/(len(clv_data)))
repeat_shoppers = clv_data[clv_data['frequency']>0]
print(repeat_shoppers.describe())

# train gamma-gamma model
gg_mod = GammaGammaFitter(penalizer_coef = 0.1)
gg_mod.fit(repeat_shoppers['frequency'], repeat_shoppers['monetary_value'])
print(gg_mod)


# On average customers make an average of 4  repeat purchases. About 37% of customers in the data have had only one transaction and are considered one-time buyers. For the Gamma-Gamma model, we only want to focus on the customers who are repeat customers to better understand the customer base. 

# In[82]:


# estimate average transaction value for each customer
clv_data['pred_txn_value'] = round(gg_mod.conditional_expected_average_profit(clv_data['frequency'], clv_data['monetary_value']), 2)
clv_data.sort_values(by = 'pred_txn_value', ascending = False).reset_index().head()


# From the output above we can see that customer 15749.0 will be expected to have an estimated value of 9967.29 in transactions.

# In[83]:


# calculate clv
clv_data['clv'] = round(gg_mod.customer_lifetime_value(
    bgf_mod,
    clv_data['frequency'],
    clv_data['recency'],
    clv_data['T'],
    clv_data['monetary_value'],
    time = 12,
    discount_rate = 0.01), 2)

clv_data.sort_values(by = 'clv', ascending = False).reset_index().head(10)


# From the output above we see that customer 18102.0 has a lifetime value of 34019.02 over the next 12 months. 

# # Association Rule Mining - Apriori Algorithm
# 
# Most retail giants need to apply data mining processes to uncover associations between products to better understand their customers. For the dataset, which is a collection of transactions, Association Rule Mining will be used to enhance product placement and to target customers with personalized promotions. For example, if x buys milk then how likely is it that this customer will also buy bread in the same transaction. Association Rule Mining are if-then statements that help uncover patterns of co-occurrence. The 'if' is the antecedent and the 'then' is the consequent. For association rules, support and confidence are two important metrics to consider. Support is the proportion of transactions in the database that contain items A and B in the antecedent and the consequent. If the support is low, that means there is insufficient evidence that items in the itemset of occurring together. Confidence is the probability that the transaction will include items A and B in the consequent given the number of times A occurs in the antecedent. Apriori algorithm will be used to extract frequent itemsets to generate association rules in this analysis. Apriori Theorem states that "if an itemset is frequent, then all of its subsets must also be frequent." The user-specified support threshold will be applied to find all frequent itemsets in the transactional database. Then, these frequent items and the minimum confidence constraint will be used to form association rules. 

# In[43]:


# group by invoice number and desription and sum of quantity
cart = uk_data.groupby(['invoice_no', 'description'])['quantity']


# In[44]:


# create one-hot encoded table that shows number of products purchased
cart = cart.sum().unstack().reset_index().fillna(0).set_index('invoice_no')

# view transaction basket
cart.reset_index().head()


# After applying a one-hot encode you will see several zeros in the data frame signifying products that were not purchased for a particular invoice number. A number greater than zero tells you how many products were sold for that particular invoice number.

# In[45]:


# create function that converts all positive values to 1 and everything else to 0
def encode_test(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


# To simplify the algorithm we want to assign values greater than zero 1 and others zero. 

# In[46]:


# apply function
cart_sets = cart.applymap(encode_test)
cart_sets.head()


# In[47]:


# form frequent itemsets
freq_itemsets = apriori(cart_sets, min_support = 0.01, use_colnames = True)


# As mentioned the Apriori algorithm requires a user-specified minimum level of support. The minimum level of support is the percentage of transactions in the database that contain an itemset.

# In[48]:


# generate association rules
assoc_rules = association_rules(freq_itemsets, metric = 'lift', min_threshold = 1)
assoc_rules.sort_values(by = 'confidence', ascending = False, inplace = True)
assoc_rules.head(25)


# Above are the association rules for 25 itemsets sorted by highest-lowest confidence value in the UK transactional database. All itemsets above have a minimum support value of 1% and a lift value of over 1. A lift value greater than 1 would mean item Y is likely to be purchased if item X is purchased. If you look at the first itemset in the above output, it states: if Herb Marker Thyme then Herb Marker Rosemary is purchased with a support value at 0.010267. This means that at least 1% of all transactions in the database have a combination of Herb Marker Thyme and Herb Marker Rosemary being purchased in the same basket. There is also 94% confidence that Herb Marker Rosemary is sold whenever Herb Marker Thyme is purchased. A lift value of 85.96 means that Herb Marker Thyme being bought has an influence on the purchase of Herb Marker Rosemary. We have enough evidence to suggest that conclude that the purchase of Herb Marker Thyme results in the purchase of Herb Marker Rosemary.

# # Recommendations
# 
# Through K-Means clustering I was able to segment the customer population for the online retailer which could help strategize promotions for specific loyalty levels resulting in personalized marketing campaigns. Through Customer Lifetime Value I was able to discover customers that are expected to have higher returns and others that are at risk of churning, which will allow for the company to focus on marketing to new customers to grow customer population. Through Association Rule Mining I was able to develop a recommendation engine that will allow the company to strategize its offers for certain products as well as product placement on the website. However, a challenge with Association Rule Mining would be a lack of personalized offers being developed because the product insights are not unique to specific customers. I would recommend the company focus on developing a rewards loyalty program that customers can sign up for and earn points based on certain products they purchase and how often. 

# ## References

# https://lifetimes.readthedocs.io/en/latest/lifetimes.html
# 
# https://lifetimes.readthedocs.io/en/latest/lifetimes.html#module-lifetimes.generate_data
# 
# https://pandas.pydata.org/docs/
# 
# https://www-users.cs.umn.edu/~kumar001/dmbook/ch5_association_analysis.pdf
# 
# http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/
# 
# http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/
# 
# https://www.blastanalytics.com/blog/rfm-analysis-boosts-sales
# 
# https://towardsdatascience.com/find-your-best-customers-with-customer-segmentation-in-python-61d602f9eee6
