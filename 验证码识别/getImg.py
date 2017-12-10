#!/usr/bin/python  
# -*- coding:utf8 -*-  
  
import os 

# 方法一
def file_name(file_dir):
	for root, dirs, files in os.walk(file_dir):
	    # print(root) #当前目录路径  
	    # print(dirs) #当前路径下所有子目录  
	    print(files) #当前路径下所有非目录子文件

# 方法二
def gci(filepath):
#遍历filepath下所有文件，包括子目录
  files = os.listdir(filepath)
  for fi in files:
    fi_d = os.path.join(filepath,fi)            
    if os.path.isdir(fi_d):
      gci(fi_d)                  
    else:
      print(os.path.join(filepath,fi_d))    


# gci('/Users/yangyibo/GitWork/pythonLean/AI/验证码识别/img/')
# file_name('/Users/yangyibo/GitWork/pythonLean/AI/验证码识别/img/')

def getlabel(len,str):
  number = []
  for item in range(0,len,1):
    number.append(int(str[item:item+1]))
  print(number)
    # item = item+1




number ='1024'


getlabel(len(number),number)


numlist=[int( number[item: item+1] ) for item in range(0, len(number), 1)]
print(numlist)  

