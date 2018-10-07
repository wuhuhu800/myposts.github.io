---
layout: post
title: 淘宝商品详情页图片抓取
categories: 爬虫
description:  python抓取淘宝商品详情页图片
keywords: 爬虫、slenium、chromedriver
tags: jekyll
---

本文主要是对淘宝详情页图片进行爬取，主要核心技术就是对淘宝反爬虫技术ajax突破，具体见代码

```Python
# 导入selenium的浏览器驱动接口
from selenium import webdriver
# 要想调用键盘按键操作需要引入keys包
from selenium.webdriver.common.keys import Keys
# 导入chrome选项
from selenium.webdriver.chrome.options import Options
from lxml import etree
import os,time,re,requests,shutil
from PIL import Image


taobao_url=input('请输入抓取地址: ')
output_=input('请输入最后图片命名: ')
# 创建chrome浏览器驱动，无头模式（超爽）
chrome_options = Options()
chrome_options.add_argument('--headless')
driver = webdriver.Chrome(chrome_options=chrome_options)

driver.get(taobao_url)#获取淘宝地址
time.sleep(3)

# 逐渐滚动浏览器窗口，令ajax逐渐加载
for i in range(1, 20):#滚动的次数
    js = "var q=document.body.scrollTop=" + str(500 * i)  # PhantomJS
    js = "var q=document.documentElement.scrollTop=" + str(500 * i)  # 谷歌 和 火狐
    driver.execute_script(js)
    #print('=====================================')
    time.sleep(3)
# 拿到页面源码
html = etree.HTML(driver.page_source)
all_img_list = []
img_group_list = html.xpath("//div[@id='J_DivItemDesc']//descendant::img/@src")


#保存抓取图片
def saveImg(savePath, imageURL, fileName):
        """存储图片方法
        Arguments:
            imageURL <string> -- 图片的url
            fileName <string> -- 用于存储的文件名
        """
        print('开始保存图片' + fileName)
        with open(savePath + '/' + fileName+ '.png', 'wb') as f:
            f.write(requests.get(imageURL).content)

#调整图片，目的为了衔接自然
def image_resize(img, w, h):
    """调整图片大小
    """
    try:
        if img.mode not in ('L', 'RGB'):
            img = img.convert('RGB')
        img = img.resize((w,h))
    except Exception as e:
        pass
    return img

#合并图片
def image_merge(images, output_dir='output', output_name='merge.jpg'):
    """垂直合并多张图片
    images - 要合并的图片路径列表
    ouput_dir - 输出路径
    output_name - 输出文件名
    """
    max_width = 0
    total_height = 0
    # 计算合成后图片的宽度（以最宽的为准）和高度
    for img_path in images:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            width, height = img.size
            if width > max_width:
                max_width = width
            total_height += height

    # 产生一张空白图
    new_img = Image.new('RGB', (max_width, total_height), 255)
    # 合并
    x = y = 0
    for img_path in images:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            width, height = img.size
            if width <max_width:
                img= image_resize(img, max_width, height)
            new_img.paste(img, (x, y))
            y += height

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = '%s/%s' % (output_dir, output_name)
    new_img.save(save_path)
    return save_path


#保存图片的位置
img_dir_path='/Users/suncan/Desktop/天下秀/pic'

if os.path.exists(img_dir_path):
    shutil.rmtree(img_dir_path)#清除之前抓的图片

if not os.path.exists(img_dir_path):
    os.makedirs(img_dir_path)

#保存抓取的每一张图片
index=0
for img in img_group_list:
    imglink = img
    if img.startswith('http') is True:
        imglink = img
        saveImg(img_dir_path, imglink, str(index))
        index += 1

#遍历存放图片的路径
images=[]
for root, dirs, files in os.walk(img_dir_path):
    for i in files:
        obpath=os.path.join(root,i)
        images.append(obpath)

#rm -rf '/Users/suncan/Desktop/天下秀/pic/.DS_Store'

if __name__ == "__main__":
    image_merge(images,output_name=output_+'.jpg')

```
参考：

[使用slenium+chromedriver实现无敌爬虫](https://blog.csdn.net/u010986776/article/details/79266448)

[使用PIL实现多张图片垂直合并](http://www.redicecn.com/html/Python/20130523/459.html)
