# Pandas 数据操作

    >>> df：
        name	key1	key2
    0	Allen	11		12
    1	Bob	    21		22
    2	Celina	31		32
### 0. 打开读取文件:
```
import pandas as pd
df = pd.read_csv('path/file.csv', usecols={'name', key1'}, dtype={'name': str, 'key1': int}, encoding='utf8')
df = df.read_excel('D:/Program Files/example.xls',sheetname=0)
```

### 1. 生成Dataframe:
```
df = pd.DataFrame(np.array([['Allen', 11, 12], ['Bob', 21, 22], ['Celina', 31, 32]]), columns=['name', 'key1', 'key2'], index=[0, 1, 2])

data = [['2', '1.2', '4.2'], ['0', '10', '0.3'], ['1', '5', '0']] 
df = pd.DataFrame(data, columns=['key1', 'key2', 'key3'])

df_empty = pd.DataFrame(columns=['key1', 'key2', 'key3'])

data = { 'row1' : [1,2,3,4], 'row2' : ['a' , 'b' , 'c' , 'd'] }
df = pd.DataFrame(data)
```

### 2. 选择特定行:
```
df = df[df['key1']==a1]
df = df[(df['key1']==a1) & (df['key2']==a2)]
df = df[(df['key1']==a1) | (df['key1']==b1)]
df.iloc[2:3, :]     
df[df['name'].isin(['Allen', 'Bob'])]
df[~df['name'].isin(['Allen', 'Bob'])]
df.query('a>0').query('0<b<2')
df.query('a>0 and 0<b<2')
```

### 3. 选择特定列:
```
df = df['key1']
df = df[['key1', 'key2']]
df.iloc[:, 2]
df[df['one'].isin(list)]
```

取某列值: df.columns.values

### 4. 分组:
```
grouped=df['key2'].groupby(df['key1'])
grouped.mean()
grouped.sum()
grouped.count()
df.groupby(['key1','key2'], mapping, axis=1)
```


mapping (dict): 对列进行进一步分组
axis(bool): 行操作还是列操作

### 5. 对group by后的内容进行操作，如转换成字典: 
```
piece=dict(list(df.groupby('key1'))) -> dict(pd.Dataframe)
dict = df.set_index('name').T.to_dict('list')
dict = df.set_index('key1')['key2'].T.to_dict()
dict = df.set_index(['key1', 'key2'])['value'].T.to_dict()
```

#### 聚合运算

计算pd.Series或Dataframe某列

```
df['key1'].agg('mean')
df['key1'].agg(['mean', 'max'])
df['key1'].agg([('Mean', 'mean'), ('Std', 'std')])
```

count, sum, mean, median, std, var, min, max, prod(非NA值的积), first(第一个非NA的值), last

### 6. Dataframe拼接

    >>> df
    	name	key1	key2
    0	Allen	11		12
    1	Bob	    21		22
    2	Celina	31		32
    
    >>> df2
    		key3	key4
    0		13		14
    1		23		24
    2		33		34
    
    >>> df3
    	name	key3	key4
    0	Allen	13		14
    1 	Bob		23		24
    2	David	43		44
    3	Ella	53		54
#### join函数: 将两个DataFrame中的不同的列索引合并成为一个DataFrame	
```
df.join(df2, on=None, how='left', lsuffix='', rsuffix='',sort=False) 
```

```
>>>
	name	key1	key2	key3	key4
0   Allen	11   	12    	13    	14
1   Bob   	21   	22    	23    	24
2  	Celina  31   	32    	33    	34
```

```
df.join(df3.set_index('name'), on='name')
```

```
>>>
	name	key1	key2	key3	key4
0   Allen	11   	12    	13    	14
1   Bob   	21   	22    	23    	24
2  	Celina  31   	32    	33    	34
```

#### merge函数: 默认how = inner
```
pd.merge(df, df3, on='name', how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)
```

```
>>>
	name	key1	key2	key3	key4
0   Allen   11   	12    	13    	14
1   Bob   	21   	22    	23    	24
```

```
pd.merge(df, df3, on='name', how='left')
```

```
>>>
	name	key1	key2	key3	key4
0   Allen   11   	12    	13    	14
1   Bob   	21   	22    	23    	24
2  	Celina  31   	32    	NaN    	NaN
```

```
pd.merge(df, df3, on='name', how='right')
```

```
>>>
	name	key1	key2	key3	key4
0	Allen	11		12		13		14
1 	Bob		21		22		23		24
2	David	NaN		NaN		43		44
3	Ella	NaN		NaN		53		54
```

```
pd.merge(df, df3, on='name', how='outer')
```

```
>>>
	name	key1	key2	key3	key4
0	Allen	11		12		13		14
1 	Bob		21		22		23		24
2	Celina	31		32		NaN		NaN
3	David	NaN		NaN		43		44
4	Ella	NaN		NaN		53		54
```

#### concat函数: 轴向连接, 单纯地把两个表拼在一起, 默认axis=0, 输出列顺序可能打乱
```
pd.concat([df, df3], ,ignore_index=True)
```

```
>>>
	key1	key2	key3	key4	name
0	11		12		NaN		NaN		Allen
1	21		22		NaN		NaN		Bob
2	31		32		NaN		NaN		Celina
3	NaN		NaN		13		14 		Allen
4	NaN		NaN		23		24		Bob
5	NaN		NaN		33		34		David
6	NaN		NaN		43		44		Ellen
```

```
pd.concat([df, df3], axis=1)
```

```
>>>
	name	key1	key2	name	key3	key4
0	Allen	11		12		Allen	13		14
1	Bob		21		22		Bob		23		24
2	Celina	31		32		David	33		34
3	NaN		NaN		NaN		Ellen	43		44
```

### 7. 数据归一化
```
df = (df - df.min()) / (df.max() - df.min())
df["A"] = df["A"].apply(np.log1p)
```

### 8. 线性回归

```
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(df['x'].values.reshape(-1, 1), df['y'])
a, b = regr.coef_, regr.intercept\_
regr.predict(df['x'].values.reshape(-1,1))
```

### 9. SQL操作

```
from pandasql import *
pysqldf = lambda q: sqldf(q, globals())
q = """select * from data WHERE GROUP BY"""
df = pysqldf(q)
```

### 10. 多个相同Dataframe拼接

```
Folder_Path = r'E:\DD1'  # 要拼接的文件夹及其完整路径，注意不要包含中文
SaveFile_Path = r'E:\Demand Data 4'  # 拼接后要保存的文件路径
SaveFile_Name = r'all2.csv'  # 合并后要保存的文件名

# 修改当前工作目录
os.chdir(Folder_Path)
# 将该文件夹下的所有文件名存入一个列表
file_list = os.listdir()

# 读取第一个CSV文件并包含表头
df = pd.read_csv(Folder_Path + '\\' + file_list[0])  # 编码默认UTF-8，若乱码自行更改

# 将读取的第一个CSV文件写入合并后的文件保存
df.to_csv(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf_8_sig", index=False)

# 循环遍历列表中各个CSV文件名，并追加到合并后的文件
for i in range(1, len(file_list)):
    df = pd.read_csv(Folder_Path + '\\' + file_list[i])
    df.to_csv(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf_8_sig", index=False, header=False, mode='a+')
```

