---
layout: post
title: CS231n Winner 2016 lecture 8 Location and Detection
categories: 机器学习
description: Location and Detection
keywords: 机器学习,CV,computer vision
tags: jekyll
---

[课程链接](https://www.youtube.com/watch?v=wFG_JMQ6_Sk&list=PLLvH2FwAQhnpj1WEB-jHmPuUeQ8mX-XXG&index=8)

**3:48 **
![](/images/lecture/snapshot/computer_vision_tasks.png)

Location and Detection big differnt is the number objects that we are finding.So in location there is sort of one object or  in general a fixed
number of objects whereas Detection we might have multiple objects,or a variable number of objects.
there are a little different but which can affect different architure.

**7:09 **
![](/images/lecture/snapshot/location_and_regression.png)

需要定位的话，一般把左上角定位坐标原点，需要4个点进行定位出框。对于定位这种方式，人们很自然的想到用回归的方法。loss，用[Euclidian loss](https://gist.github.com/kgrm/e0741af9fda6ee7b04871d10f6a1d811)是一个标准的选择


**8:26 **
![](/images/lecture/snapshot/recipe_for_classfication_location.png)
对于location 可以使用[L2 loss](https://segmentfault.com/a/1190000010338204) 和ground tree boxes


**10:08 **
![](/images/lecture/snapshot/attach_the_regressor_head.png)

可以在最后一个卷积层或者最后一个全连接层进行连接回归的算法的头

**3:48 **
![](/images/lecture/snapshot/sliding_window.png)

定位的另外一种思路就是slide window,[overfeat](https://blog.csdn.net/whiteinblue/article/details/43374195)



**20:41 **
![](/images/lecture/snapshot/imagenet_classficaiton_location.png)

微软2015年的残差算法还是很强大的

**26:04 **
![](/images/lecture/snapshot/detection_as_classfication.png)



**30:31 **
![](/images/lecture/snapshot/region_proposals.png)
对于dectation的问题是，有很多区域需要try，加上使用CNN这样计算量就非常大了。解决方法就是通过某种方式猜一些区域，这种方法就是region proposals，这种方法的优势就是快，尽管牺牲掉了准确率。
可以直接找到一些blobby，进行识别？？？


**31:36 **
![](/images/lecture/snapshot/selective_search.png)
受到region proposal的blobby启发，selective search通过将附近相同颜色的合并，然后再将相同颜色生成框各种不同框，之后再做侦测



**20:41 **
![](/images/lecture/snapshot/imagenet_classficaiton_location.png)
