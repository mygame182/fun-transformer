# 笔记3


Decoder主要介绍了Transformer中编码器的工作流程、结构（组成部分）。本章节详细介绍了多头自注意力机制。章节阐述如何计算缩放点积注意力，多头（Multi-Head）和自注意力运算（Self Attention）的具体含义，前馈全连接层的计算和作用等。本章节还详细对比了多头自注意力和多头注意力、自注意力运算和交叉注意力运算。内容详实，逻辑清晰。具体细节还有待后续查看，可能有些内容还需要补充。


## Transformer的结构

![图片描述](./images/C3images11.png)


## Encoder 详解


## 问题解答

### 1. 什么是多头注意力中的Concat操作？
在 Transformer 模型中，Concat（Concatenate，拼接） 是一种将多个向量或矩阵沿着特定维度连接在一起的操作。它的核心目的是合并不同来源的信息，让模型能够综合利用这些信息进行后续计算。

pytorch中利用torch.cat函数 即可拼接所有头的输出
```python
# 假设有 8 个注意力头，每个头输出维度 64
head1_output = ...  # 形状 [batch_size, seq_len, 64]
head2_output = ...  # 形状 [batch_size, seq_len, 64]
...
head8_output = ...  # 形状 [batch_size, seq_len, 64]

# 拼接所有头的输出（沿最后一个维度）
concatenated = torch.cat([head1_output, head2_output, ..., head8_output], dim=-1)
# 形状变为 [batch_size, seq_len, 8×64=512]

# 线性投影降维回 512 维
final_output = linear_layer(concatenated)  # 形状 [batch_size, seq_len, 512]
```


对比相加和拼接操作
|操作|特点|应用场景|
|-----|-----|----|
|Add（相加）|直接融合信息，不改变维度，依赖残差设计|	残差连接、门控机制（如 LSTM）|
|Concat（拼接）|保留所有原始信息，增加维度，需后续处理（如线性投影）|多头注意力、多特征源合并|


