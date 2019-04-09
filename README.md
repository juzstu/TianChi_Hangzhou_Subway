# TianChi_Hangzhou_Subway
全球城市计算AI挑战赛
# 比赛链接： https://tianchi.aliyun.com/competition/entrance/231708/introduction?spm=5176.12281925.0.0.507071376UQKMJ

## A榜：Rank4
## B榜：Rank7
## C榜：Rank4

## 代码说明：
- station_matrix.m: 根据车站训练的word2vec文件, 主要体现的是各车站的连接信息，具体思路见[队友neuronblack的博客分享](https://neuronblack.github.io/2019/04/04/%E5%A4%A9%E6%B1%A0%E5%85%A8%E7%90%83AI%E8%AE%A1%E7%AE%97%E6%8C%91%E6%88%98%E8%B5%9B%E6%80%9D%E8%B7%AF%E5%88%86%E4%BA%AB/)。
- 模型1，基于周历史趋势建模，即当前时间对应的上周时间。
- 模型2，基于天历史趋势建模。
- 然后对两个模型预测的结果进行加权融合。
因为考虑到春运的因素，利用地铁进行通勤的人员会出现较大程度的减少，所以我们对模型预测的结果进行了适当衰减，即乘衰减系数如0.96。
