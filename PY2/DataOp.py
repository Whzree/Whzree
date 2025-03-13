#!/usr/bin/env python
# -*- coding:UTF-8 -*-
'''
@Project:pythonProject1
@File:DataOp.py
@IDE:pythonProject1
@Author:whz
@Date:2025/3/13 14:59

'''

import pandas as pd


#读取文件
examre = pd.read_csv('./result.csv',encoding='gbk')
#print(examre.head())

#只留考核考试

column_name = "考核"
examre = examre[examre[column_name] == "考试"]
#重置索引
examre = examre.reset_index(drop=True)
#print(examre["考核"])
#去掉学号或者课程号为空的行
examre = examre.dropna(subset = ['学号','课程号'])
#print(examre.iloc[7])
#print(examre["成绩"])


"""
unique_values = examre['成绩'].unique()
print("成绩列的唯一值:", unique_values)
a = '\u3000'
b = '成绩'
result = examre[examre[b] == a]
print(result)
"""

#成绩也是字符串
# 替换空值为 0
examre['成绩'] = examre['成绩'].replace(['\u3000', ''], '0')

# 转换为数值类型
examre['成绩'] = pd.to_numeric(examre['成绩'], errors='coerce')

# 填充 NaN 值为 0
examre['成绩'] = examre['成绩'].fillna(0)

examre = examre[['学号', '课程号', '成绩']]
#4通一个学生，冗余成绩保留最高成绩 按学号分组
examre = examre.loc[examre.groupby(["学号","课程号"])['成绩'].idxmax()]
examre = examre.reset_index(drop=True)

#print(examre.head())

#5对每科成绩采用z-score方法进行规范化
examre = examre[['学号', '课程号', '成绩']]
def z_score_normalization(x):
    if (x.std() == 0):
        return x*0
    else:
        return (x - x.mean()) / x.std()
def safe_equal_depth_binning(x):
    try:
        return pd.qcut(x, q=4, labels=['A','B','C','D'], duplicates='drop')
    except:
        return pd.Series(['D']*len(x), index=x.index, dtype='category')

examre['成绩'] = examre.groupby('课程号')['成绩'].transform(z_score_normalization)
examre.to_csv('cjb1.csv',index=False)
examre['等级'] = examre.groupby('课程号')['成绩'].transform(safe_equal_depth_binning)
examre.to_csv('cjbresult1.csv',index=False)
#将规范化后的每一科成绩按照等深分箱法离散化，分数从高到低分别表示为A,B,C,D。四个分箱层

#因为有些std求得为0


