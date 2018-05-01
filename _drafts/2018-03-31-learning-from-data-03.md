---
layout: post
title: learning from data 3rd lession notes
categories: 机器学习
description:  The Linear Model I
keywords: 机器学习,learning from data,
tags: jekyll
---

## The Linear Model I


本节课的内容
![](https://user-images.githubusercontent.com/21167490/38158222-d8c3260c-34c2-11e8-8e02-99fdede27764.png)

![](https://user-images.githubusercontent.com/21167490/38158262-7618e072-34c3-11e8-93f1-46edc2501fef.png)


**12:49**

![](/images/lecture/snapshot/input_representation.png)  
输入值：
图片为16X16的像素，所以X的输入共有256个，再加上常数X_0共257个,对应的W的向量也有257个。这样仅仅一个小图就有这么多输入，运算量都非常大。因此需要建立Feature(特征)，找到最有用的信息，减少输入量。识别数字特征：intensity和
symmetic，这样一样257个特征一下子就缩减为3个(包括X_0)，这样W向量也变为3个，大大减少了运算量。

**14：10**
![](images/lecture/snapshot/illustration_of_feature.png)  
对比1和5，从图中可以看到，5是大部分属于红色取样，这些红色区域属于intensity的坐标
而1属于蓝色区域，蓝色区域属于symmetric坐标

PLA：perceptron learning algorithm

** 20:30 **
![](images/lecture/snapshot/pocket_algorithm.png)
pocket algorithm 属于贪心算法


**39:43 **  
![](images/lecture/snapshot/minimize_weight.png)
Weight计算，可以通过X，y计算得出

**44:27 **
![](images/lecture/snapshot/linear_regrssion_algorithm.png)

线性回归
W=pseudo-inverse with y

**50：36 **
![](images/lecture/snapshot/linear_regression_boundary)
对于二分类问题，如果直接求W的话，从头训练，可能需要很多周折
可以用linear_regression先得出一组比较好的W，然后再通过这组W作为initial值，再做二分类，效果会更好


** **
![](images/lecture/snapshot/)



** **
![](images/lecture/snapshot/)
