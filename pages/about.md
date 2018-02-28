---
layout: page
title: About
description: 达者为师
keywords: Woody, 阿虎
comments: true
menu: 关于
permalink: /about/
---

我是阿虎，也叫Woody，相信：

**学无先后，达者为师**

**无他，唯手熟尔**


## 联系

{% for website in site.data.social %}
* {{ website.sitename }}：[@{{ website.name }}]({{ website.url }})
{% endfor %}

## Skill Keywords

{% for category in site.data.skills %}
### {{ category.name }}
<div class="btn-inline">
{% for keyword in category.keywords %}
<button class="btn btn-outline" type="button">{{ keyword }}</button>
{% endfor %}
</div>
{% endfor %}
