---
layout: post
title: learning from data 2nd lession notes
categories: 机器学习
description:  Is learning feasible
keywords: 机器学习,learning from data,
tags: jekyll
---


## Is learning feasible
![](https://user-images.githubusercontent.com/21167490/38029690-2b05781c-32c9-11e8-874e-481049233158.png)

**12:22**
![](https://user-images.githubusercontent.com/21167490/38132843-58036712-343f-11e8-90e8-0e58ddcb2b84.png)
样本概率v是 全体概率μ 吗，很显然不是

**19：20**
![](https://user-images.githubusercontent.com/21167490/38133280-29826e4a-3441-11e8-9081-2a09c676f982.png)
epison 代表容忍度

**25：48**
![](https://user-images.githubusercontent.com/21167490/38133890-926da31e-3443-11e8-95dd-41d7573366ee.png)
1、该公式N和epsion是有效值  
2、公式的右边不依赖与μ  
3、如果要epsion越小，N就得越大  
4、v可以推出μ，但是v是由μ决定的  

**29：49**
![](https://user-images.githubusercontent.com/21167490/38134323-09469210-3445-11e8-8484-9ef709938f67.png)
类比机器学习，x是bin里的marbles，如果假设h和真实f关系，通过对于h(x)和y进行比对，如果h(x)=f(x)，那么bin标记为green，如果不等，bin标记为red。现在问题是μ和f都是unkown的。

**53：20**
![](https://user-images.githubusercontent.com/21167490/38136443-e4dff96e-3450-11e8-93bf-814fc479265e.png)
**55：06**
![](https://user-images.githubusercontent.com/21167490/38136508-78b5a030-3451-11e8-8c49-0eef4af83197.png)

E_in(g)：选定g模型（最终模型）之后，样本的发生概率  
E_out(g): 选定g模型（最终模型）之后，全体的发生概率  
** **
![]()
