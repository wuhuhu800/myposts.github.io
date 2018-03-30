---
layout: post
title: learning from data 1st lession notes
categories: 机器学习
description:  The essence of machine learning
keywords: 机器学习,learning from data,
tags: jekyll
---

## The essence of machine learning
- A pattern exists
- We cannot pin it down mathematically
- We have data on it


![](https://user-images.githubusercontent.com/21167490/37871060-0aae1902-3017-11e8-9c4f-769c093a8eef.png)
建模，现有的问题，存在一个从x到y的函数f，我们通过历史数据(x1,y1)...(xn,yn)
找出一个函数g，使得g越来越像f

![](https://user-images.githubusercontent.com/21167490/37871121-d1f0723e-3018-11e8-854d-bd85925b9fe0.png)

梳理一下机器学习的流程图：  
1、目标是从x到y ，找到函数      
2、有一组训练数据(x1,y1)...(xn,yn)       
3、通过算法找到g，使得g逼近f      
这里重点是存在一组假设H(这个应该就是超参数？？？)，找到最贴近的假设，例如可能是linear model，可能是 support vector machine等等这些假设。   
但是这组假设是没有downside的，因为你可以穷尽所有的可能去找Hypothesis
但是有一个upside，这个upside可以告诉我们怎么改进，什么是正确的方向

![](https://user-images.githubusercontent.com/21167490/37871271-c1c09962-301c-11e8-8210-74d9016bb84c.png)
solution的解决部分
1、The Hypothesis Set    
这是由我们可感知(percetron)的模型，例如神经模型、SVM...    
H是超参数集    
h是其中一个可能的参数   
g是最终集   
2、The learning Algorithm    
这是由我们我们可感知(percetron)的算法，例如反向传播算法、一元二次方程...   
1和2分构成learning model

![](https://user-images.githubusercontent.com/21167490/38022418-03f1e7fe-32b2-11e8-861e-52a7b7aa5212.png)

有三种机器学习方式  
1、supervised learning
有输入，正确的输出(起到label或者flag的作用)
![](https://user-images.githubusercontent.com/21167490/38023488-03aea5cc-32b5-11e8-8464-e9608367dd4a.png)
2、Unsupervised learning
有输入，但没有明确的输出
![](https://user-images.githubusercontent.com/21167490/38023490-03ff173c-32b5-11e8-9598-431b10196625.png)  
3、Reinforcement learning
有输入，还有一定的输出(起到辅助作用)
![](https://user-images.githubusercontent.com/21167490/38023492-04532048-32b5-11e8-8fbb-c30b107632bf.png)
