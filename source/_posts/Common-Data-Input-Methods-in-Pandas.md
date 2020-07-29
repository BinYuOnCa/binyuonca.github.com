---
title: Common Data Input Methods in Pandas
date: 2020-07-24 17:06:04
tags:
 - Pandas

Category:
  - Python

---


This notebook will demo the most common data input methods in Pandas.

- [read_csv](#read_csv)
- [read_excel](#read_excel)
- [read_html](#read_html)
- [read_sql](#read_sql)

<!-- more -->


## Read CSV file <a id="read_csv"></a> 

One of the most common functions/methods for pandas to populate a DataFrame is read data from a [comma-seprated values (CSV)](https://en.wikipedia.org/wiki/Comma-separated_values) file.

Pandas provides a native funtcion read_csv to perform this task. Thought it supports rich features, the most common/simple way is pass it the file name and it will read the file into DataFrame:

```python
df = pd.read_csv(file_name)
df.head()
```


```python
import pandas as pd 
```


```python
file_name = 'iris.csv'
df = pd.read_csv(file_name)
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



If we do not have columns name in the first row in the file, we can specify them in parameter names:


```python
file_name = 'Running.csv'
df = pd.read_csv(file_name, names = ['DateTime','HeartRate','Pace'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DateTime</th>
      <th>HeartRate</th>
      <th>Pace</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-02-22T02:24:14Z</td>
      <td>100</td>
      <td>2.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-02-22T02:24:16Z</td>
      <td>104</td>
      <td>2.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-02-22T02:24:18Z</td>
      <td>104</td>
      <td>2.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-02-22T02:24:20Z</td>
      <td>107</td>
      <td>2.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-02-22T02:24:22Z</td>
      <td>108</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>
</div>



Some other common/useful parameters:

* **sep**: str, default ‘,’   
   useful when data file use other seperater like space,'|', '\t', etc.  
   
* **skiprows**: list-like, int or callable, optional  
   useful when the file has some explaining rows at the top  
   
* **skipfooter**: int, default 0  
   useful when file has some extra info at the end of the file  
   
* **nrows**: int, optional  
   useful when you just want read partical rows of a large file  
   

Although we have many parameters to deal with date parsing, personally I prefer to read the date as is in first  step, following some specific function to parse them seperately. My faviourate is to employ **lambda** function to do the trick.


We can also specify a URL as the file name as below.


```python
url = 'http://cocl.us/Geospatial_data'
geo_info = pd.read_csv(url)
```


```python
geo_info.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>



## Read Excel File <a id="read_excel"></a>


The function <font style="color:blue">read_excel()</font > provides the capability to read excel files into pandas DataFrame. Similiar with the function **read_csv**, the most basic use-case is passing the file name of the Excel file, and the <span style="color:blue">sheet_name</span>, will read the file into DataFrame:

```python
df = pd.read_excel(file_name, 'Sheet1')
df.head()
```


```python
df = pd.read_excel('SYS_DD.xlsx','Sheet1')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sys_dd_id</th>
      <th>dd_category_cd</th>
      <th>dd_value</th>
      <th>dd_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>101</td>
      <td>GPRS_DATA_TYPE</td>
      <td>R1</td>
      <td>数据上传</td>
    </tr>
    <tr>
      <th>1</th>
      <td>102</td>
      <td>GPRS_DATA_TYPE</td>
      <td>R2</td>
      <td>保留</td>
    </tr>
    <tr>
      <th>2</th>
      <td>103</td>
      <td>GPRS_DATA_TYPE</td>
      <td>R3</td>
      <td>校准上传</td>
    </tr>
    <tr>
      <th>3</th>
      <td>104</td>
      <td>GPRS_DATA_TYPE</td>
      <td>R4</td>
      <td>版本上传</td>
    </tr>
    <tr>
      <th>4</th>
      <td>105</td>
      <td>GPRS_DATA_TYPE</td>
      <td>R5</td>
      <td>传感器限值上传</td>
    </tr>
  </tbody>
</table>
</div>



## Read HTML page <a id="read_html"></a>


Function <font style="color:blue">read_html()</font >  is not as common as <font style="color:blue">read_csv/read_excel</font>, but sometimes it's ready easy the job when you want to fetch table-like data from a website.

This function searches for '\<table>' elements and only for "\<tr>" and "\<th>" rows and "\<td>" elements within each "\<tr>" or "\<th>" element in the table. "\<td>" stands for “table data”.

Please note that the function will always return a list of DataFrame or it will fail, it will <font style="color:red">NOT</font> return an empty list.
  
    
```python
    
try:
    dfs = pd.read_html(url)
except Exception as e:
    print('Error:{}'.format(e))
    pass  
```


```python
url = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
try:
    dfs = pd.read_html(url)
    print('Found {} tables in URL:{}'.format(len(dfs), url))
except Exception as e:
    print('Error:{}'.format(e))
```

    Found 3 tables in URL:https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M


Now we can safely check what's inside these DataFrames, and find out which one is we want.


```python
for idx, df in enumerate(dfs):
    print('DataFrame[{}]:{}'.format(idx, df.shape))

dfs[0].head()    
```

    DataFrame[0]:(180, 3)
    DataFrame[1]:(4, 18)
    DataFrame[2]:(2, 18)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1A</td>
      <td>Not assigned</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M2A</td>
      <td>Not assigned</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
    </tr>
  </tbody>
</table>
</div>



Lets wrap <font style="color:blue">read_html</font> to a function, and check the following scenario:   
(1) URL not found   
(2) No table in specific URL



```python
def demo_read_html(url):
    try:
        dfs = pd.read_html(url)
        print('Found {} tables in URL:{}'.format(len(dfs), url))
        return dfs 
    
    except Exception as e:
        print('Error:{}'.format(e))    
        return None 
    
```

(1) URL not found   



```python
url = 'https://en.wikipedia.org/wiki/xxxxx'
ret = demo_read_html(url)

```

    Error:HTTP Error 404: Not Found


(2) No table in speific URL


```python
url = 'https://maps.google.com'
ret = demo_read_html(url)

```

    Error:No tables found


## Read Data from DB <a id="read_sql"></a>


Database is the most import data source for many applications. Most companies persistence their business data in varies Databases. In this notebook, we will demostrate how to use pandas function <font style="color:blue">read_sql</font> to fetch data from PostgresSQL database.

The usage of function <font style="color:blue">read_sql</font> is very simpliar with these ones described above:

```python 
df = pd.read_sql(' select * from oridata ', conn)
df.head()

``` 

Assume that we already set up the PostgresSQL Database with data populated in table **oridata**. Also the PostgresSQL adapter **Psycopg** installed as well.



```python
import psycopg2 as pg
```


For data security, the best practise is never hard code the user name, password in the code. Instead of, we employed a simple trick here, to save the credencial in a local file, and make sure this file is not visible to anyone else. More advanced security solution is out of the scope of this demo, will discuss it seperately.



```python
import pickle 

with open('db.info','rb') as input:
    dbConnInfo = pickle.load(input)
    
dbConnInfo.database
```




    'zlydb'



Now we can connect to the Database '**zlydb**' and read table **oridata** from the Database.


```python
conn = pg.connect( database = dbConnInfo.database, user = dbConnInfo.user, password = dbConnInfo.password )

```


```python
df = pd.read_sql(' select * from oridata ', conn)
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stockid</th>
      <th>dt</th>
      <th>jbcj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SH600000</td>
      <td>2018-01-02</td>
      <td>8954.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SH600000</td>
      <td>2018-01-03</td>
      <td>10182.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SH600000</td>
      <td>2018-01-04</td>
      <td>7726.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SH600000</td>
      <td>2018-01-05</td>
      <td>8126.44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SH600000</td>
      <td>2018-01-08</td>
      <td>9226.89</td>
    </tr>
  </tbody>
</table>
</div>



### Appendix

Below please find the python code to prepare the data info credentical file: 

```python 
class dbinfo:
    database = "your DB Name"
    user="your DB user name"
    password="your password "
    
with open('db.info', 'wb') as output:
    pickle.dump(dbinfo, output)
```    

You may also add the file into git ignore file to avoid it being pushed into the public repo:

```shell
$ echo db.info >> .gitignore
```
