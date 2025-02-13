# 笔记2



##  **补充：矩阵乘法的计算复杂度**

假设有两个矩阵：
- **矩阵 A**：维度为 $ (m \times n) $
- **矩阵 B**：维度为 $ (n \times p) $

那么 **矩阵乘法 $ C = A \times B $** 的计算复杂度是多少？

矩阵乘法的定义如下：
$$
C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}
$$
- 每个元素 $ C_{ij} $ 需要 **n 次乘法和 n-1 次加法**。
- **C 矩阵的维度为 $ (m \times p) $**，所以总共需要计算 **$ m \times p $ 个元素**。
- **乘法次数**：$ m \times p \times n $
- **加法次数**：$ m \times p \times (n - 1) $ （通常忽略加法，重点分析乘法）
- **时间复杂度**：$ O(mnp) $

### **特殊情况**
- **方阵乘法**（如果 $ m = n = p $）：计算复杂度为 **$ O(n^3) $**。
- **向量点积**（如果 $ m = p = 1 $）：计算复杂度为 **$ O(n) $**。


### **复杂度优化**
经典矩阵乘法为 **$ O(mnp) $**，但在特殊情况下，可以使用更快的算法：
- **Strassen 算法**（适用于方阵，降低到 **$ O(n^{2.81}) $**）
- **Coppersmith-Winograd 算法**（目前最优 **$ O(n^{2.37}) $**）
- **GPU 并行计算**（利用批量矩阵运算加速）

---

### **矩阵乘法计算复杂度总结**
| 乘法类型 | 计算复杂度 |
|----------|------------|
| 一般矩阵乘法 ($ m \times n $ 和 $ n \times p $) | $ O(mnp) $ |
| 方阵乘法 ($ n \times n $) | $ O(n^3) $ |
| Strassen 算法 | $ O(n^{2.81}) $ |
| Coppersmith-Winograd 算法 | $ O(n^{2.37}) $ |
| 向量点积 ($ d $-维向量) | $ O(d) $ |

**另外：尽管有些时候数学计算复杂度相同，我们还需要考虑内存访问的复杂度，这里往往会有几倍的差距**<p>
以计算 $A^TB$和 $AB^T$为例子。计算 $A^TB$ 需要额外存储 $A^T$ ，大规模运算时会占用更多的内存，计算 $AB^T$ 则不需要。

---











## 问题解答
### 1. 什么是软注意力和硬注意力？<p>
- **软注意力（Soft Attention）** 是一种平滑、连续的机制，适用于大多数需要在多个输入元素之间进行加权计算的任务，且可以通过标准的梯度下降方法优化。在软注意力机制中，模型对**所有输入**进行计算，给每个输入分配一个权重（即注意力权重）。<p>
- **硬注意力（Hard Attention）** 是一种离散的、非可微的机制，通常用于那些需要明确选择某些输入元素的任务，但由于其不可微的特性，训练过程更加复杂，需要强化学习等方法来进行优化。硬注意力在处理时**选择**某些输入元素，直接“关注”它们，而忽略其他输入。<p>
一言以蔽之，硬注意力在每一步操作时，只选择一个或少数几个特定的输入元素，而软注意力对所有输入进行加权平均。详情看[软注意力和硬注意力](https://github.com/RINZERON/fun-transformer/blob/main/docs/chapter2/%E8%BD%AF%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%92%8C%E7%A1%AC%E6%B3%A8%E6%84%8F%E5%8A%9B.md)


### 2. 什么是“自注意力”（Self-Attention）机制？<p>





### 3. 掩码多头自注意力有什么用？


### 4. 什么是受限的自注意力运算（restricted Self-Attention）?

就是设定窗口的自注意力运算，跟局部注意力机制类似。计算 Attention 时仅考虑每个输出位置附近窗口的 r 个输入。这带来两个效果：计算复杂度降为 $O(r \times n \times d)$ 最长学习距离降低为 $r$，因此需要执行 $O(\frac{n}{r})$ 次才能覆盖到所有输入。（考虑 $n$ 个 key 和 $n$ 个 query 两两点乘，维数为 $d$ ）


### 5. **先安装python，移植到conda环境中可能出现pip install location位置不对，可能路径依赖问题？**

重新用Conda创造一个干净的环境，做为项目的专用python环境。
在CMD中输入，
>conda create --name env_name python=3.9/3.8...

之后安装相应的库。（记得在对应的环境中）
>pip install package_name<p>
>conda install  package_name

**注意**：在VSCode中的终端选择正确的内核。如果想要在jupyter中安装，则ipynb文件单元格中应该是
>%pip install package_name

安装相应库后，要重启jupyter内核，程序才能找到包。

### 6. **numpy与gensim包冲突怎么解决？**
在CMD输入

>pip install --upgrade gensim

自动更新gensim包，并且会检查numpy包是否符合要求，卸载并安装合适的numpy包。

在CMD输入
>pip check

可查看有无冲突的包。
>No broken requirements found.

### 7. **怎么在VSCode中查看jupyter的变量值？**

安装扩展（ctrl+shift+x）Data Wrangler。
VSCode提示安装相应的包，选择确定即可。或者我们自行在终端输入（安装相应的包）。
>pip install package_name

Data Wrangler 的核心依赖包包括：
- pandas、numpy：数据处理。
- scikit-learn：机器学习。
- matplotlib、seaborn：数据可视化。
- ipykernel、jupyter：交互式环境支持。

安装这些包后，我们可以在 VSCode 中充分利用 Data Wrangler 的功能进行数据分析和清洗。如果遇到问题，请检查包是否安装正确，并确保 VSCode 使用了正确的 Python 环境。
