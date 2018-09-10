# 第三届阿里云安全算法挑战赛

[竞赛链接](https://tianchi.aliyun.com/competition/introduction.htm?raceId=231668)

这个版本相对于 [v0.1](https://github.com/beader/3rd_security/tree/v0.1)，训练逻辑和效果不变，但做了一些优化:

- 使用 Tensorflow TFRecord 格式存储原始数据
- 中间步骤的脚本支持管道操作, 具体可以查看脚本的 --help
- 训练部分使用 TFRecord，训练速度更快，占用内存变少

## 背景

此竞赛是阿里云举办的第三届安全算法挑战赛，这一次竞赛的主题是恶意文件分类。

本题目提供的数据来自 windows 可执行程序 经过沙箱程序模拟运行后的 API 指令序列，经过了脱敏处理。

其中恶意文件的类型有

|文件类型|label|
|---:|:---|
|正常文件	| 0 |
|感染型病毒| 1 |
|木马程序	| 2 |
|挖矿程序	| 3 |
|DDOS木马| 4 |
|勒索病毒	| 5 |

## 数据说明

### 3rd_security_train.zip

字段说明

|字段|类型|解释|
|:---|:---|:---|
|file_id|bigint|文件编号|
|label|bigint|文件标签，0-正常/1-勒索病毒/2-挖矿程序/3-DDoS木马/4-蠕虫病毒/5-感染型病毒|
|api|string|文件调用的API名称|
|tid|bigint|调用API的线程编号|
|return_value|string|API返回值|
|index|string|线程中API调用的顺序编号|

数据预览

```bash
$ unzip -p 3rd_security_train.zip | head
file_id,label,api,tid,return_value,index
0,0,GetSystemTimeAsFileTime,2644,0,0
0,0,NtAllocateVirtualMemory,2644,0,1
0,0,NtFreeVirtualMemory,2644,0,2
0,0,NtAllocateVirtualMemory,2644,0,3
0,0,NtAllocateVirtualMemory,2644,0,4
0,0,NtAllocateVirtualMemory,2644,0,5
0,0,NtAllocateVirtualMemory,2644,0,6
0,0,SetUnhandledExceptionFilter,2644,0,7
0,0,LdrLoadDll,2644,0,8
...
```

需要注意的点:

1. 返回值经过脱敏处理，它并不代表返回值是一个数字
2. 一个文件可能起了很多个进程，某个进程的 api 序列可能很长，这里做过截断处理，把序列长度超过 5000 的部分进行了截断删除
3. index 是单个文件在沙箱执行时的全局顺序，由于沙箱执行时间有精度限制，所以会出现一个 index 上出现同线程或者不同线程都在执行多次 api 的情况。index 可以保证 tid 内部的顺序，但不保证连续

简单的统计

```bash
# 训练集行数
$ unzip -p 3rd_security_train.zip | wc -l
409631050

# 训练集文件数
$ unzip -p 3rd_security_train.zip | tail -n +2 | cut -f 1 -d, | uniq | wc -l
116624

# label 的分布情况
$ unzip -p 3rd_security_train.zip | tail -n +2 | cut -f 1,2 -d, | uniq | cut -f 2 -d, | sort | uniq -c
 111545 0
    287 1
    744 2
    598 3
     53 4
   3397 5

# 测试集文件数
$ unzip -p 3rd_security_test.zip | tail -n +2 | cut -f 1 -d, | uniq | wc -l
53093

```

训练集一共 11万+ 个文件样本，共4亿个 API 调用记录。label分布很不平衡，其中 4 类别的样本数极少

|字段|类型|解释|
|:---|:---|:---|
|file_id|bigint|文件编号|
|label|bigint|文件标签，0-正常/1-勒索病毒/2-挖矿程序/3-DDoS木马/4-蠕虫病毒/5-感染型病毒|
|api|string|文件调用的API名称|
|tid|bigint|调用API的线程编号|
|return_value|string|API返回值|
|index|string|线程中API调用的顺序编号|


## 赛题理解

假设某个文件有以下的 api 调用顺序，采集精度为10ms，这里忽略  api 的 return value

```bash
index                   0   1   2       3   4   5           6   7
timestamp(ms)           0   10  20  30  40  50  60  70  80  90  100
tid             1001    a   a,a         a   a   a           c   a
                1002        b   c           d                   b
                1003    f               f                   f
```


那么将生成如下数据文件

```bash
api	tid	index
a	1001	0
a	1001	1
a	1001	1
a	1001	3
a	1001	4
a	1001	5
c	1001	6
a	1002	7
b	1002	1
c	1002	2
d	1002	4
b	1002	7
f	1003	0
f	1003	3
f	1003	6
```

## 评测指标

官方说明采用 logloss 计算:

![3](http://orxuqm7p7.bkt.clouddn.com/3.png?imageView2/2/h/60)

但实际上，logloss 公式应该是:

![4](http://orxuqm7p7.bkt.clouddn.com/2018-08-30-4.png?imageView2/2/h/70)


参考 [fast.ai wiki - logloss](http://wiki.fast.ai/index.php/Log_Loss)

不清楚最终评分是按照哪个来？

## 模型思路

### 原始序列向量化

这个题目类似一个 NLP 问题，你把每个文件的调用序列，当成一篇文章来看，类似 NLP 的文本分类。可以尝试的做法有构建  n-gram tfidf 特征，然后用 Random Forest , GBDT 等算法构建多分类模型。但在这里我想尝试使用深度学习的方法，避免人工特征工程。因为时间关系以及提交次数的限制，无法尝试太多的模型架构，也没有对模型做很好的调优工作，最终该模型的 logloss 在 0.09 左右，在排行榜上只能处在中游水平，不过我相信这种方法有一定的潜力，未来如果我们能够自行搜集更多的样本，尝试更复杂的结构，相信其表现可以提升很多。

前面说到这个问题其实是 sequence → class 的一个问题，但仔细分析，它又有自己不一样的地方。

对于一个文本来说，词与词之间是连续的，比如

```bash
index   0    1    2     3    4    5 6
word    I like this movie very much !
```

由于采集精度的问题，譬如精度只有10ms，在上表的例子中，index 下有两个 api 无法区分其先后。除此之外，一个文件可能会启动多个线程

因此站在文件的层面出发，我们看到的是这样一种序列

```bash
{a, f} -> {a, a, b} -> {c} -> {} -> {a, f} -> {a, d} -> {a} -> {} -> {} -> {c, f} -> {a, b}
```

由于 index 只包含了先后关系，并没有间隔长度的关系，因此实际上我们能看到的序列是这种形式

```bash
{a, f} -> {a, a, b} -> {c} -> {a, f} -> {a, d} -> {a} -> {c, f} -> {a, b}
```

因此实际上文本类型的 sequence 是上述 sequence 的一个特例，上述 sequence 是 a sequence of sets，文本类型的 sequence 相当于每个 set 中只包含一个元素

```bash
{I} -> {like} -> {this} -> {movie} -> {very} -> {much} -> {!}
```

对于 text sequence 来说，我们可以用 [One-Hot Encoding](https://en.wikipedia.org/wiki/One-hot) 对词进行编码，变成一个 2D Matrix

![text_ohe](http://orxuqm7p7.bkt.clouddn.com/text_ohe.png?imageView2/2/h/300)

相同的，我们可以把 API sequence 也变成 One-Hot Encoding

![api_seq_ohe](http://orxuqm7p7.bkt.clouddn.com/api_seq_ohe.png?imageView2/2/h/300)

注: 这种表示方法，忽略了在同一个 index 下，相同 api 被调用多次的情况


### CNN 网络结构

网络的设计参考了这篇文章 [《Understanding how Convolutional Neural Network (CNN) perform text classification with word embeddings》](http://www.joshuakim.io/understanding-how-convolutional-neural-network-cnn-perform-text-classification-with-word-embeddings/)。

区别在于我在最初的地方加另一个 maxpool 层

![cnn_structure](http://orxuqm7p7.bkt.clouddn.com/cnn_structure.png?imageView2/2/h/500)

### 训练过程
考虑到 class 0 的样本占据了大多数，并且较难的部分其实是 class 1 ~ class 5 之间的分类问题。因此在每一个 epoch 中，我们随机抽取一部分 class 0 的样本，其余 class 保持固定。

## 使用方法

将数据文件放入 `./data` 目录下


```bash
$ unzip -p ./data/3rd_security_train.zip | python encode_values.py -c api --vocab-file value_mappings.csv \
    | python to_tfrecords.py -o train:0.8 valid:0.2 --with-label --compress

$ unzip -p ./data/3rd_security_test.zip | python encode_values.py -c api --vocab-file value_mappings.csv \
    | python to_tfrecords.py -o test --compress
```

训练

```bash
$ python train.py --train-file train.tfrecords.gz  --valid-file valid.tfrecords.gz
```

预测

```bash
$ python predict.py --test-file test.tfrecords.gz --model-file ./logs/cnn_xxx/cnn_xxx.h5 --submission-file submissions.csv
```

## 后续的一些反思

1. 关于评分标准，按照阿里云说明文档中，按照 logloss 进行评估，但给出的公式并不是 logloss 的公式，有可能出现的情况是，官方按照自己的那个公式做评分，而训练时的 loss 是按照 logloss 去做最优化的，这里可能会吃亏。
2. 这次比赛并没有尝试普通的Random Forest或者GBDT去做，直接用了一个最简单的 CNN 去做，我觉得这个 baseline model 的 performance 还凑合，毕竟这里面没有任何的人工特征工程在里面。在初赛阶段大约能达到 0.09+，后续如果优化网络结构以及做一些fine-tune，也许有一定的潜力。
3. 这里简单使用了 one-hot encoding，因为还没有想到一个学习 embedding 的一个好方法，因为和文本不一样，这种类型的 sequence 中每个 element 是一个 set。

