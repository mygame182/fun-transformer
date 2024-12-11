# fun-transformer
本课程涵盖了Transformer的核心组成部分，包含Transformer编码器和解码器的实现。
课程结合代码解读知识点，同时通过形象生动的例子，为学习者理解Transformer模型提供参考和借鉴。
实践项目无需依赖任何深度学习框架，完全从零开始，使用基础的Numpy等科学计算库实现Transformer，旨在深化学习者对模型本质的理解与掌握。
最后，使用Transformer模型实现在机器翻译任务中的应用，加深对模型的理解


## 课程大纲
| 章节  | 内容 | 代码实现|
| ------------- | ------------- |------------- |
| 第一章 | 引言[introduction](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter1/introduction.md) |词嵌入[低维映射到高维](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter1/低维映射到高维.ipynb)|
| 第二章 | Transformer简述[Transformer](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter2/Transformer.md) |  相对位置向量[相对位置向量](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter2/相对位置向量.ipynb)  |
| 第三章 | Encoder结构[Encoder](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter3/Encoder.md)   | 交叉注意力[Cross-Attention](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter3/CrossAttention.ipynb)      |
|第四章   |Decoder结构[Decoder](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter4/Decoder.md)| bert[apply-bert](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter4/apply-bert.ipynb)、gpt[apply-gpt](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter4/apply-gpt.ipynb)      |
|第五章 |项目实践| 机器翻译项目案例、Transformer结构拆解、使用 NumPy 和 SciPy 实现通用注意力机制|

## 目录
第一章 引言[introduction](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter1/introduction.md)
- 1. 序列到序列（Seq2Seq）模型概述
- 2. Encoder-Decoder模型概述
- 3. Attention 的提出与影响

第二章 Transformer简述[Transformer](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter2/Transformer.md)
- 1. Attention 机制
- 2. Transformer概述
- 3. Transformer vs CNN vs RNN
- 4. 输入嵌入(Input Embedding)
- 5. Multi-Head Attention vs Multi-Head Self-Attention

第三章 Encoder结构[Encoder](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter3/Encoder.md)
-  1. 编码器(Encoder)
- 2. 多头自注意力(Multi-Head Self-Attention)
- 3. 交叉自注意力(Cross Attention)
- 4. Cross Attention 和 Self Attention 主要的区别

第四章 Decoder结构[Decoder](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter4/Decoder.md)
- 1. 解码器(Decoder)
- 2. 掩码(Mask)
- 3. 模型的训练与评估
- 4. 高级主题和应用
- 5. [Tokenization](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter4/Tokenization.md)
  
第五章 Project
- 1. 项目案例[实践项目](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter5/实践项目.ipynb)
- 2. 使用NumPy和SciPy实现通用注意力机制[使用NumPy和SciPy实现通用注意力机制](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter5/%E4%BD%BF%E7%94%A8%20NumPy%20%E5%92%8C%20SciPy%20%E5%AE%9E%E7%8E%B0%E9%80%9A%E7%94%A8%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.ipynb)
- 3. 一键运行Transformer板块[Transformer组件实现](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter5/Transformer组件实现.ipynb)
- 4. Multi-head attention[多头注意力机制](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter5/多头注意力机制.ipynb)
- 5. Self attention[自注意力机制实现](https://github.com/datawhalechina/fun-transformer/blob/main/docs/chapter5/自注意力机制实现.ipynb)
## 参与贡献

- 如果你想参与到项目中来欢迎查看项目的 [Issue]() 查看没有被分配的任务。
- 如果你发现了一些问题，欢迎在 [Issue]() 中进行反馈🐛。
- 如果你对本项目感兴趣想要参与进来可以通过 [Discussion]() 进行交流💬。

如果你对 Datawhale 很感兴趣并想要发起一个新的项目，欢迎查看 [Datawhale 贡献指南](https://github.com/datawhalechina/DOPMC#%E4%B8%BA-datawhale-%E5%81%9A%E5%87%BA%E8%B4%A1%E7%8C%AE)。

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :----| :---- | :---- |
| 罗清泉 | 项目负责人 |  |



## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

*注：默认使用CC 4.0协议，也可根据自身项目情况选用其他协议*
