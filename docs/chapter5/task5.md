# 笔记5



### 自注意力机制实现.ipynb代码笔记





### 实践项目.ipynb代码笔记



``` python

'''
要在GPU上运行，需要把所有数据加载到GPU上，包括中间的计算结果，
模型 Transformer 和 nn.LayerNorm 都要加载到GPU上
损失函数，优化器等等
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):    # 遍历每句话
      enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]    # Encoder_input 索引
      dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]    # Decoder_input 索引
      dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]    # Decoder_output 索引
      enc_inputs.extend(enc_input)
      dec_inputs.extend(dec_input)
      dec_outputs.extend(dec_output)
    return (torch.LongTensor(enc_inputs).to(device), 
           torch.LongTensor(dec_inputs).to(device), 
           torch.LongTensor(dec_outputs).to(device))    #把数据集加载到GPU上


'''
## 前馈神经网络
输入inputs ，经过两个全连接层，得到的结果再加上 inputs （残差），再做LayerNorm归一化。LayerNorm归一化可以理解层是把Batch中每一句话进行归一化。
调用nn.LayerNorm默认是在CPU上运行，我们要添加.to(device)将模型加载到GPU上
'''

class FF(nn.Module):
    def __init__(self):
        super(FF, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):    # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)   # [batch_size, seq_len, d_model]


'''
省略 中间所有的中间变量添加.to(device)
'''

'''
# 定义网络
'''


model = Transformer().to(device)    #模型加载到GPU上
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 占位符 索引为0.
optimizer_SGD = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
optimizer_Adam = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
optimizer = optimizer_SGD


'''
SGD 随机梯度下降 更适合图像等密集数据
Adam Adam算法 更适合NLP任务
'''


'''
# 训练Transformer
'''
epoch_num = 50 # 训练20轮 不一定要越多越好 但太少肯定不行 关键看loss的变化

time_start = time.time()

for epoch in range(epoch_num):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        # print("输入：",enc_inputs, dec_inputs, dec_outputs)
        enc_inputs, dec_inputs, dec_outputs = enc_inputs, dec_inputs, dec_outputs    # [2,5] [2,5] [2,5]
        # print("又输入：",enc_inputs, dec_inputs, dec_outputs)
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))  # outputs: [batch_size*tgt_len, tgt_vocab_size]
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

time_end = time.time()
print('训练完成! 耗时:', time_end - time_start)




```