# Pytorch基本操作

### 1.1 新建Tensor

```
import torch
# 默认torch.FloatTensor
# 传入数据用tensor，传入形状用FloatTensor
# 小写的tensor接受数据，
# 大写的Tensor()或者FloatTensor()接受的是shape，数据的维度
a = torch.Tensor([[1, 2], [3, 4], [5, 6]])
a = torch.DoubleTensor([[1, 2], [3, 4], [5, 6]])
a = torch.LongTensor([[1, 2], [3, 4], [5, 6]])
# 整型
a = torch.ShortTensor([[1, 2], [3, 4], [5, 6]])
a = torch.IntTensor([[1, 2], [3, 4], [5, 6]])
a = torch.zeros((3, 2))
a = torch.randn((3, 2))
y = torch.ones(1)
# 变量
# Only Tensors of floating point dtype can require gradients
x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad = True)
y = 2 * x
# 默认torch.tensor([1])
y.backward(torch.ones((3, 2)))
torch.zeros(*sizes, out=None, ..)# 返回大小为sizes的零矩阵 

torch.zeros_like(input, ..) # 返回与input相同size的零矩阵

torch.ones(*sizes, out=None, ..) #f返回大小为sizes的单位矩阵

torch.ones_like(input, ..) #返回与input相同size的单位矩阵

torch.full(size, fill_value, …) #返回大小为sizes,单位值为fill_value的矩阵

torch.full_like(input, fill_value, …) 返回与input相同size，单位值为fill_value的矩阵

torch.arange(start=0, end, step=1, …) #返回从start到end, 单位步长为step的1-d tensor.

torch.linspace(start, end, steps=100, …)  #返回从start到end, 间隔中的插值数目为steps的1-d tensor

torch.logspace(start, end, steps=100, …) #返回1-d tensor ，从10^start到10^end的steps个对数间隔
```

### 1.2 Tensor操作

```
a.size()
a[0, 1] = 10
```

### 1.3 数据类型转化

```
b = a.numpy()
c = torch.from_numpy(b)
d = c.float()
```

# Pytorch示例

## 1. 全连接神经网络

### 1.1 模型

```
import torch
N_batch, dimension_input, dimension_hidden, dimension_output =  64, 1000, 100, 10
x = torch.randn(N_batch, dimension_input)
y = torch.randn(N_batch, dimension_output)
model = torch.nn.Sequential(
		torch.nn.Linear(dimension_in, dimension_hidden),
		torch.nn.Relu(),
		torch.nn.Linear(dimension_hidden, dimension_output))
```

### 1.2 参数优化(不使用Optimizer)

1. Loss反向传播前清零**model**梯度
2. 参数优化前使用no_grad暂停梯度更新

```
loss_function = torch.nn.MSELoss(reduction = 'sum')
learning_rate = 1e-4
for t in range(500):
	y_pred = model(x)
	loss = loss_function(y_pred, y)
	# print(loss.item)
	# 反向传播之前清零梯度
	model.zero_grad()
	loss.backward()
	# 参数更新前停止梯度传播
	with torch.no_grad():
		for param in model.parameters:
			param -= learning_rate * param.grad
```

### 1.3 参数优化(使用Optimizer)

1. Loss反向传播前清零**optimizer**梯度
2. 反向传播之后更新参数权重

```
loss_function = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optimum.SGD(model.parameters(), lr = 1e-4, momentum = 0.9)
for t in range(500):
	y_pred = model(x)
	loss = loss_function(y_pred, y)
	# print(loss.item)
	# 清零梯度, 反向传播, 更新权重
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
```

### 1.4 保存、加载模型

保存整个模型:

```
torch.save(model,PATH)
model = torch.load(PATH)
```

仅保存学习到的参数:

```
# state_dict是在定义了model或optimizer之后pytorch自动生成的
# 只有那些参数可以训练的layer才会被保存到模型的state_dict中,如卷积层,线性层等等
print(optimizer.state_dict())
torch.save(model.state_dict(), PATH)
# 加载模型
new_model = TheModelClass(*args, **kwargs)
new_model.load_state_dict(torch.load(PATH))
new_model.eval()
```

## 2. 线性回归示例

### 2.1 准备数据

```
import torch
x = torch.tensor([[1.000], [2.000], [3.000], [4.000], [5.000]], requires_grad = True).cuda()
y = torch.tensor([[2.000], [4.000], [6.000], [8.000], [10.000]], requires_grad = True).cuda()
```

### 2.2 定义模型、损失函数和优化器

```
class LR(torch.nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        out = self.linear(x)
        return out
model = LR().cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

### 2.3 训练模型  

```
n_epochs = 1000
for epoch in range(n_epochs):
    inputs = x.cuda()
    target = y.cuda()
    out = model(inputs)
    loss = criterion(out, target)
    optimizer.zero_grad() # 每步之前对优化器归零
    loss.backward() # 反向求导
    optimizer.step() # 更新参数
    print(loss.data)
```

### 2.4 测试结果

```
model.eval()
predict = model(x).data.cpu().numpy()
```

## 3. CNN

$$
卷积核[5*5] = \left[ 
\matrix{  
1 & 0 & -1 & 0 & 1 \\  \
1 & 0 & -1 & 1 & 0 \\  \
-1 & 1 & 0 & 1 & 1 \\ \
1 & 0 & -1 & 0 & 1 \\  \
1 & 0 & -1 & 1 & 0 \\  
} 
\right]
$$

```
import torch.nn.functional as f
# 每一个模型需定义两个函数, __init__(声明模型结构)和forward(梯度正向传播方式)
class Net(nn.module):
	def __init__(self):
		# super函数继承nn.Module的__init__()
		super(Net.self).__init__()
		# 输入向量:(1, 1, 32, 32), (batch_size=1, channel=1, length=32, width=32)
		# 卷积层1, 6个5 * 5尺寸卷积核	
		self.conv1 = nn.Conv2d(1, 6, 5)
		# 卷积层2, 16个5 * 5尺寸卷积核	
		self.conv2 = nn.Conv2d(6, 16, 5)
		# 池化后: 6*14*14 → 16*10*10 → 16*5*5
		# 线性层1: 400 → 120
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		# 线性层2: 120 → 84
		self.fc2 = nn.Linear(120, 84)
		# 线性层3(输出层): 84 → 10 
		self.fc3 = nn.Linear(84, 10)
	def forward(self, x):
	    # 输入向量:(1, 1, 32, 32)
		x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2)) # (1, 6, 14, 14)
		x = f.max_pool2d(f.relu(self.conv2(x)), (2, 2)) # (1, 16, 5, 5)
		x = x.view(-1, self.num_flat_features(x)) # (1, 16 * 5 * 5)
		x = f.relu(self.fc1(x)) # (1, 120)
		x = f.relu(self.fc2(x)) # (1, 84)
		x = self.fc3(x) # 最终结果: (1, 10)
		return x
	def num_flat_features(self, x):
	    # size: [16, 5, 5]
		size = x.size([1:])
		num_features = 1
		for s in size:
			num_features *= s
		# num_features = 400
		return num_features 
```

## 4. RNN

### 4.1 GRU 

#### 4.1.1 原理

​	GRU（Gate Recurrent Unit）是循环神经网络（Recurrent Neural Network, RNN）的一种。和LSTM（Long-Short Term Memory）一样，也是为了解决长期记忆和反向传播中的梯度等问题而提出来的。

​	GRU和LSTM在很多情况下实验效果与LSTM相似，但是更易于计算。

​	GRU的输入输出结构与普通的RNN相同, 有一个当前的输入 $x^t$，和上一个节点传递下来的隐状态 $h^{t-1}$，这个隐状态包含了之前节点的相关信息。结合二者，GRU会得到当前隐藏节点的输出 $y^t$ 和传递给下一个节点的隐状态 $h^t$ 。
$$
GRU(h^{t-1},x^t)\rightarrow h^t,y^t
$$
<img src="https://raw.githubusercontent.com/Canadasunyan/pics/master/v2-49244046a83e30ef2383b94644bf0f31_r.png" alt="v2-49244046a83e30ef2383b94644bf0f31_r" style="zoom: 33%;" />

首先，我们先通过上一个传输下来的状态$h^{t-1}$和当前节点的输入$x^t$来获取两个门控状态。如下图2-2所示，其中 $r$为控制重置的门控（reset gate）， $z$为控制更新的门控（update gate）。

<div align="center"><img src="C:\Users\ROG\Desktop\v2-5b805241ab36e126c4b06b903f148ffa_r.jpg" alt="v2-5b805241ab36e126c4b06b903f148ffa_r" style="zoom: 33%;" />
​	运算定义：

$$
Hadamard \ Product: \\
x \odot y= \left[
\matrix{
  x_{11}*y_{11} & x_{12}*y_{12} & ... & x_{1n}*y_{1n} \\
  x_{21}*y_{21} & x_{22}*y_{22} & ... & x_{2n}*y_{2n} \\
  ... & ... & ... & ...\\
  x_{m1}*y_{m1} & x_{m2}*y_{m2} & ... & x_{mn}*y_{mn} 
}
\right]
$$


$$
Matrix Addition: \\
x \oplus y= \left[
\matrix{
  x_{11}+y_{11} & x_{12}+y_{12} & ... & x_{1n}+y_{1n} \\
  x_{21}+y_{21} & x_{22}+y_{22} & ... & x_{2n}+y_{2n} \\
  ... & ... & ... & ...\\
  x_{m1}+y_{m1} & x_{m2}+y_{m2} & ... & x_{mn}+y_{mn} 
}
\right] \\
$$
​	得到门控信号之后，首先使用重置门控来得到"重置"之后的数据：
$$
r=sigmoid(W^r\left[
\matrix{
  x^{t} \\
  h^{t-1}}
\right]) \in (0,1) \\
z=sigmoid(W^z\left[
\matrix{
  x^{t} \\
  h^{t-1}}
\right]) \in (0,1) \\
h^{t-1'}=h^{t-1} \odot r
$$
​	这里的$h^{'}$主要是包含了当前输入的$x^t$数据。有针对性地对$h^{'}$添加到当前的隐藏状态，相当于”记忆了当前时刻的状态“。类似于LSTM的选择记忆阶段:
$$
h^{'}=tanh(W\left[
\matrix{
  x^{t} \\
  h^{t-1'}}
\right]) \\
$$
​	最后介绍GRU最关键的一个步骤，可以称之为”更新记忆“阶段。在这个阶段，我们同时进行了遗忘了记忆两个步骤。我们使用了先前得到的更新门控。$z$ 越大，信息更新越快:
$$
h^t=(1-z) \odot h^{t-1} + z \odot h^{'}
$$
​	GRU很聪明的一点就在于使用了同一个门控就同时可以进行遗忘和选择记忆 (LSTM则要使用多个门控), 这里的遗忘$z$和选择$(1-z)$是联动的。也就是说，对于传递进来的维度信息，我们会进行选择性遗忘，则遗忘了多少权重，我们就会使用包含当前输入的$h^{'}$中所对应的权重进行弥补$(1-z)$ , 以保持一种”恒定“状态。

#### 4.1.2 Pytorch实现

1. Embedding: 将词映射为特定维度的向量

```
embed = torch.nn.Embedding(n_vocabulary,embedding_size)
```

2. Tokenize: 标准化

```
['I am a boy.','I am very lucky.','Fuck you all!']
```

​		(1) 小写化:

```
[['i','am','a','boy','.'],['i','am','very','lucky','.'],['fuck','you','all','!']]
```

​		(2) 向量化:

```
batch = [[1,2,3,4,5], [1,2,6,7,5], [8,9,10,11]]
```

​		(3) 结尾加EOS (=-1):

```
batch = [[1,2,3,4,5,-1], [1,2,6,7,5,-1], [8,9,10,11,-1]]
lens = [6,6,5]
```

​		(4) 使用padding (=0) 补齐:

```
batch = [[1,2,3,4,5,-1], [1,2,6,7,5,-1], [8,9,10,11,-1,0]]
```

​		(5) 转换为batch序列:

```
batch = list(itertools.zip_longest(batch,fillvalue=PAD))
# fillvalue就是要填充的值，强制转成list
batch = [[1,1,8],[2,2,9],[3,6,10],[4,7,11],[5,5,-1],[-1,-1,0]]
batch=torch.LongTensor(batch)
```

3. 使用建立了的embedding直接通过batch取词向量:

```
embedding_size = 6
tensor([[[-0.2699,  0.7401, -0.8000,  0.0472,  0.9032, -0.0902],
         [-0.2699,  0.7401, -0.8000,  0.0472,  0.9032, -0.0902],
         [ 0.1146, -0.8077, -1.4957, -1.5407,  0.3755, -0.6805]],

        [[-0.2675,  1.8021,  1.4966,  0.6988,  1.4770,  1.1235],
         [-0.2675,  1.8021,  1.4966,  0.6988,  1.4770,  1.1235],
         [-0.0387,  0.8401,  1.6871,  0.3057, -0.8248, -0.1326]],

        [[-0.0387,  0.8401,  1.6871,  0.3057, -0.8248, -0.1326],
         [-0.3745, -1.9178, -0.2928,  0.6510,  0.9621, -1.3871],
         [-0.6739,  0.3931,  0.1464,  1.4965, -0.9210, -0.0995]],

        [[-0.2675,  1.8021,  1.4966,  0.6988,  1.4770,  1.1235],
         [-0.7411,  0.7948, -1.5864,  0.1176,  0.0789, -0.3376],
         [-0.3745, -1.9178, -0.2928,  0.6510,  0.9621, -1.3871]],

        [[-0.3745, -1.9178, -0.2928,  0.6510,  0.9621, -1.3871],
         [-0.3745, -1.9178, -0.2928,  0.6510,  0.9621, -1.3871],
         [ 0.2837,  0.5629,  1.0398,  2.0679, -1.0122, -0.2714]],

        [[ 0.2837,  0.5629,  1.0398,  2.0679, -1.0122, -0.2714],
         [ 0.2837,  0.5629,  1.0398,  2.0679, -1.0122, -0.2714],
         [ 0.2242, -1.2474,  0.3882,  0.2814, -0.4796,  0.3732]]],
       grad_fn=<EmbeddingBackward>)
# tensor.size = [seq_len,batch_size,embedding_size]
```

4. 取词向量放入GRU

```
# 这里的input_size就是词向量的维度，hidden_size就是h的维度，这两个一般相同就可以
# n_layers是GRU的层数,多层为RMLP
# 并不需要指定时间步数，也即seq_len，这是因为，GRU和LSTM都实现了自身的迭代。
import torch.nn as nn
gru = nn.GRU(input_size=50, hidden_size=50, n_layers=1)
# 将3个词映射为50维向量
embed = nn.Embedding(3, 50)
# batch = 'i love you' & 'eat the apple'
x = torch.LongTensor([[0, 1, 2], [3, 4, 5]])
x = list(itertools.zip_longest(x,fillvalue=PAD))
x_embed = embed(x)
```

```
>>> x
[[0,3], [1,4], [2,5]]
>>> x_embed.size()
torch.Size([3, 2, 50])
```

```
# input: [seq_len, batch, input_size] (batch_first = False)

# input: [batch, seq_len, input_size] (batch_first = True)

# h0: [n_layers* n_directions, batch, hidden_size]

# output: [seq_len, batch, num_directions * hidden_size]

# h1: [n_layers * n_directions, batch, hidden_size]

gru = nn.GRU(input_size = 3, hidden_size = 10, n_layers = 2)
input = x_embed
# 单向GRU, n_directions = 1
h0 = torch.randn(2 * 1, batch = 2, hidden_size = 10)
output, h1 = gru(input,h0)
```

```
>>> print(output.shape,h1.shape)
torch.Size([3, 2, 10]) torch.Size([2, 2, 10])
```

# Pytorch调参

## 1. 定义自动求导函数

​	首先介绍`forward`函数，此函数必须:

1. 接受一个context ctx作为第一个参数，之后可以传入任何参数;
2. ctx可以利用**save_for_backward**来保存tensors，在backward阶段可以进行获取。forward里定义了前向传播的路径. 自己定义的Function中的forward()方法，所有的Variable参数将会转成tensor, 因此这里的input也是tensor．在传入forward前，autograd engine会自动将Variable unpack成Tensor。
3. 可以有任意多个输入、任意多个输出，但是输入和输出必须是Variable。

​    之后介绍backward函数，此函数必须:

1. 接受一个context ctx作为第一个参数，第二个参数**grad_output**(Tensor)存储forward后output的梯度;
2. return的结果是对应于forward里的各个**input**的梯度。
3. **saved_tensors**: 传给forward()的参数，在backward()中会用到。

​    ctx.needs_input_grad作为一个boolean型的表示也可以用来控制每一个input是否需要计算梯度，e.g., ctx.needs_input_grad[0] = False，表示forward里的第一个input不需要梯度，若此时我们return时这个位置的梯度值表示为None即可。

### 1.1 Exponential

​	Backward计算:
$$
y = e^{x} \\
$$

$$
\frac{\partial output}{\partial x}=\frac{\partial output}{\partial y}*\frac{\partial y}{\partial x} = grad\_output*e^{x}
$$

```
class Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.exp(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        return grad_output * torch.exp(input)

exp = Exp()
x1 = torch.tensor([3., 4.], requires_grad=True)
x2 = exp.apply(x1)
y2= exp.apply(x2)
y2.sum().backward()
print(x1.grad)
```

### 1.2 ReLu

$$
y =max\{0, x\} \\
\begin{align}
\frac{\partial output}{\partial x}=
\frac{\partial output}{\partial y}*\frac{\partial y}{\partial x} \\
=grad\_output\ \ or \ \  0
\end{align}
$$

```
import torch
# 所有的求导函数都继承于torch.autograd.Function
class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # clamp: x -> min(max(min_value, x), max_value)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # 调出保存的input
        input, = ctx.saved_tensors
        # grad_input = grad_output or 0
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # To apply our Function, we use Function.apply method. We alias this as 'relu'.
    relu = MyReLU.apply

    # Forward pass: compute predicted y using operations; we compute
    # ReLU using our custom autograd operation.
    y_pred = relu(x.mm(w1)).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
```

### 1.3 Linear Function

$$
y = xW^{T}+b \\
$$

$$
\frac{\partial output}{\partial x}=\frac{\partial output}{\partial y}*\frac{\partial y}{\partial x} = grad\_output*W \\
$$

$$
\frac{\partial output}{\partial W}= \frac{\partial output}{\partial y}*\frac{\partial y}{\partial W}=grad\_output^T*x \\
$$

$$
\frac{\partial output}{\partial b}= 
\frac{\partial output}{\partial y}*\frac{\partial y}{\partial b}=
grad\_output^*1^T \\
$$

```
import torch.autograd.Function as Function
class LinearFunction(Function):
　  # 创建torch.autograd.Function类的一个子类
    # 必须是staticmethod
    @staticmethod
    # 第一个是ctx，第二个是input，其他是可选参数。

    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias) # 将Tensor转变为Variable保存到ctx中
        output = input.mm(weight.t())  # torch.t()方法，对2D tensor进行转置
        # output = w.t * x + b
        if bias is not None:
        	# unsqueeze(0) 扩展处第0维
            # expand_as(tensor)等价于expand(tensor.size()), 将原tensor按新的size扩展
            output += bias.unsqueeze(0).expand_as(output)　
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        # grad_output为反向传播上一级计算得到的梯度值
        input, weight, bias, = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None
        # 分别代表输入,权值,偏置三者的梯度
        # 判断三者对应的Variable是否需要进行反向求导计算梯度
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight) # 复合函数求导，链式法则
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)　# 复合函数求导，链式法则
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias
```

## 2. 调参

### 2.1 调节学习率

#### 2.1.1 学习率为常数的学习器:

```
optimizer=torch.optim.Adam(model.parameters(),lr=0.01) 
```

#### 2.1.2 手动定义学习率

```
def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    lr *= (0.1 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
optimizer = optim.SGD(params = model.parameters(), lr=10)
for epoch in range(10):
    adjust_learning_rate(optimizer, epoch, lr_init)
    lr = optimizer.param_groups[0]['lr']
```

#### 2.1.3 等间隔调整学习率 **StepLR**

​	每过step_size个epoch, learning_rate *= 0.9

```
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.9)
model = net.train(model, loss_function, optimizer, scheduler, num_epochs = 100)
for epoch:
	scheduler.step()
```

#### 2.1.4 按需调整学习率 MultiStepLR

​	按设定的间隔调整学习率。适合后期调试使用，观察 loss 曲线，为每个实验定制学习率调整时机。

```
# 每到一个milestone, 减小学习率gamma倍
milestones=[30,80,120]
torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
```

#### 2.1.5 指数衰减调整学习率 ExponentialLR

```
lr=lr∗gamma∗∗epochlr=lr∗gamma∗∗epoch
torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
```

#### 2.1.6 LambdaLR

​	将每个参数组的学习率设置为初始lr乘以给定函数, 当last_epoch=-1时, 将初始lr设置为lr

```
# 根据epoch计算出与lr相乘的乘数因子为epoch//10的值
lambda = lambda epoch:epoch // 10 
# 根据epoch计算出与lr相乘的乘数因子为0.95 ** epoch的值
lambda = lambda epoch:0.95 ** epoch 
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda)
```

#### 2.1.7 余弦退火调整学习率

​	以余弦函数为周期，并在每个周期最大值时重新设置学习率。以初始学习率为最大学习率

```
optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4) 
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max = (epochs // 9) + 1)
```

#### 2.1.8 自适应调整学习率ReduceLROnPlateau**

​	mode为min, loss不下降学习率乘以factor, max则反之

```
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.9)
```

### 2.2 选择optimizer

#### 2.2.1 SGD

```
opt_SGD = torch.optim.SGD(model.parameters(), lr=LR)
```

#### 2.2.2 动量法

​	mini-batch SGD训练算法，虽然这种算法能够带来很好的训练速度，但是在到达最优点的时候并不能够总是真正到达最优点，而是在最优点附近徘徊。另一个缺点就是这种算法需要我们挑选一个合适的学习率，当我们采用小的学习率的时候，会导致网络在训练的时候收敛太慢；当我们采用大的学习率的时候，会导致在训练过程中优化的幅度跳过函数的范围，也就是可能跳过最优点。我们所希望的仅仅是网络在优化的时候网络的损失函数有一个很好的收敛速度同时又不至于摆动幅度太大。

​	所以Momentum优化器刚好可以解决我们所面临的问题，它主要是基于梯度的移动指数加权平均。假设在当前的迭代步骤第 tt 步中，那么基于Momentum优化算法可以写成下面的公式:
$$
\begin{eqnarray*}
v_{dW}=βv_{dw}+(1−β)dW \tag{1} \\
v_{db}=βv_{db}+(1−β)db \tag{2} \\
W=W−αv_{dW} \tag{3} \\
b=b−αv_{db} \tag{4} \\
\end{eqnarray*}
$$

```
opt_Momentum = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.8)
```

#### 2.2.3 Adagrad

​	AdaGrad、RMSProp、Adam都属于Per-parameter adaptive learning rate methods（逐参数适应学习率方法）：之前的方法是对所有的参数都是一个学习率，现在对不同的参数有不同的学习率。

​	Adagrad的一个缺点是：在深度学习中单调的学习率被证明通常过于激进且过早停止学习。

```
# Assume the gradient dx and parameter vector x
cache += dx**2
x += ‐ learning_rate * dx / (np.sqrt(cache) + eps)
torch.optim.Adagrad(net_Adagrad.parameters(),lr=Learning_rate)
```

#### 2.2.4 RMSProp

​	**Root Mean Square Prop**进一步优化损失函数在更新中存在摆动幅度过大的问题，并且进一步加快函数的收敛速度，RMSProp算法对权重 W 和偏置 b 的梯度使用了微分平方加权平均数:
$$
\begin{eqnarray*}
{dW}^2 ={dW_{(1)}}^2+{dW_{(2)}}^2+...+{dW_{(n)}}^2 \tag{1}\\
s_{dW}=βs_{dw}+(1−β){dW}^2 \tag{2} \\
s_{db}=βs_{db}+(1−β)db^2 \tag{3} \\
W=W−α\frac{dW}{\sqrt {s_{dW} + \epsilon}} \tag{4} \\
b=b−α\frac{db}{\sqrt {s_{db} + \epsilon}} \tag{5} \\
\end{eqnarray*}
$$

```
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
```

#### 2.2.5 Adam

​	Adam（Adaptive Moment Estimation）算法是将Momentum算法和RMSProp算法结合起来使用的一种算法，所使用的参数基本和上面讲的一致，在训练的最开始我们需要初始化梯度的累积量和平方累积量:
$$
\begin{eqnarray*}
s_{dW}=s_{dW}=v_{dW}=v_{dW}=0 \tag{1}\\
v_{dW}=βv_{dw}+(1−β)dW \tag{2} \\
v_{db}=βv_{db}+(1−β)db \tag{3} \\
s_{dW}=βs_{dw}+(1−β){dW}^2 \tag{4} \\
s_{db}=βs_{db}+(1−β)db^2 \tag{5} \\
W=W−α\frac{v_{dW}}{\sqrt s_{dW} + \epsilon} \tag{6} \\
b=b−α\frac{v_{db}}{\sqrt s_{db} + \epsilon} \tag{7} \\
\end{eqnarray*}
$$

```
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
```

# 源码细节

### 1.1 nn.Module

​	Module类是所有神经网络模块的基类，Module可以以树形式包含别的Module，也就是网络定义中经常使用的子网络嵌套。

#### * 1.1.1 \_\_init\_\_() 

​	使用OrderedDict是有序的字典，也就是键-值对的插入是按照次序进行的。self.\_parameters用于存储网络的参数；self.\_buffers用于存储不需要优化器进行更新的变量；self._backward\_hooks和 self.\_forward_hooks分别是前向和反向的钩子，用于获取网络中间层输入输出；self.\_state_dict_xxx表示状态字典，用于存储和参数值加载等。在创建网络时，要实现该方法，通过super.\_\_init\_\_()方法将init里的表达式继承下来:

    def __init__(self):
        """
        Initializes internal Module state, shared by both nn.Module and ScriptModule.
        """
    
        torch._C._log_api_usage_once("python.nn_module")
        self.training = True
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._state_dict_hooks = OrderedDict()
        self._load_state_dict_pre_hooks = OrderedDict()
        self._modules = OrderedDict()
#### * 1.1.2 forward() 

​	forward()函数用于前向传播的定义, 在这里没有具体实现，是需要在各个子类中实现的，如果子类中没有实现就会报错raise NotImplementedError:

```
def forward(self, *input):
	r"""Defines the computation performed at every call.
    Should be overridden by all subclasses.
    raise NotImplementedError
```

#### 1.1.3 register_buffer()

​	为那些不是parameter但是需要存储的变量注册缓存，比如BN中的mean。不是parameter表示这个量不需要被optimizer更新，也就是不需要训练。函数中大量的if-else主要用于判别输入tensor和name的类型，从而确定是否进行注册。可以在\_\_init\_\_()中使用self.register_buffer()并在实例化后使用self.named_buffers()查看已经注册的buffer：

```
def register_buffer(self, name, tensor):
    r"""Adds a persistent buffer to the module.
    This is typically used to register a buffer that should not to be
    considered a model parameter. For example, BatchNorm's ``running_mean``
    is not a parameter, but is part of the persistent state.
    Buffers can be accessed as attributes using given names.
    Args:
        name (string): name of the buffer. The buffer can be accessed
            from this module using the given name
        tensor (Tensor): buffer to be registered.
    Example::
        >>> self.register_buffer('running_mean', torch.zeros(num_features))
    """
    # 防止初始化之前就注册
    if '_buffers' not in self.__dict__:
        raise AttributeError("cannot assign buffer before Module.__init__() call")
    elif not isinstance(name, torch._six.string_classes):
        raise TypeError("buffer name should be a string. "
                        "Got {}".format(torch.typename(name)))
    # 名称检查，防止为空
    elif '.' in name:
        raise KeyError("buffer name can't contain \".\"")
    elif name == '':
        raise KeyError("buffer name can't be empty string \"\"")
    elif hasattr(self, name) and name not in self._buffers:
        raise KeyError("attribute '{}' already exists".format(name))
    # 只能注册Tensor类型    
    elif tensor is not None and not isinstance(tensor, torch.Tensor):
        raise TypeError("cannot assign '{}' object to buffer '{}' "
                        "(torch Tensor or None required)"
                        .format(torch.typename(tensor), name))
    else:
        self._buffers[name] = tensor
```

#### 1.1.4 register\_parameter()

​	注册parameter和注册buffer不同，parameter是需要进行训练更新的，注册的parameter和网络定义的卷积和全连接等的weight性质相同，可以通过self.named_parameters()查看。

​	函数同样会做一些格式和存在与否的判断，进而报错:

```
def register_parameter(self, name, param):
        r"""Adds a parameter to the module.
        The parameter can be accessed as an attribute using given name.
        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter): parameter to be added to the module.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")
        # 省略那些类型重复的检查
        elif param.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.".format(name))
        else:
            self._parameters[name] = param
```

​	可以在\_\_init\_\_()中使用self.register_parameter()并在实例化后使用self.named_buffers()查看已经注册的buffer:

```
def __init__():
	self.register_parameter(name="test param", param=nn.Parameter(torch.Tensor([2])))

net = Net()
print(net.parameters())
for param in net.named_parameters():
    print(param)
```

#### 1.1.5 add_module()

​	用于给模型添加新的操作，下面采用它给网络添加新的操作，需要注意的是Module采用的是OrderedDict，操作的添加必须是按照顺序的：

```
def add_module(self, name, module):
    r"""Adds a child module to the current module.
    The module can be accessed as an attribute using the given name.
    Args:
    name (string): name of the child module. The child module can be
    accessed from this module using the given name
    module (Module): child module to be added to the module.
    """
    # 此处省略名称和类型检查
    self._modules[name] = module
```

​	效果等同于直接添加一层:

```
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.add_module("second linear", module=nn.Linear(1, 2))

    def forward(self):
        return None

net = Net()
for param in net.modules():
    print(param)
```

#### 1.1.6 apply()

​	递归过程，对当前module的所有第一代子module施加func函数，这个func函数是自定义但是并非随意定义，fn中的操作必须是module类所能满足的。源码中给出的是参数初始化的例子，首先需要定义一个用于参数初始化的func，然后使用module.apply(fn):

```
def func(param):
    if isinstance(param, nn.Linear):
        param.weight.data.fill_(1000)
    return True
    
net = Net()
net.apply(fn=func)
print(net.linear.weight)
# tensor([[1000.]], requires_grad=True)
```

#### 1.1.7 cuda() & cpu() & to()

​	将数据在cpu和gpu之间切换，类似的还有数据类型的切换float()和double():

```
devie = torch.device('cuda')
model.to(device)
```

```
def cuda(self, device=None):
    r"""Moves all model parameters and buffers to the GPU.
    Arguments:
        device (int, optional): if specified, all parameters will be
            copied to that device
    Returns:
        Module: self
    """
    return self._apply(lambda t: t.cuda(device))
    
def cpu(self):
    r"""Moves all model parameters and buffers to the CPU.
    Returns:
        Module: self
    """
    return self._apply(lambda t: t.cpu())
  
def to(self, *args, **kwargs):
    r"""Moves and/or casts the parameters and buffers.
    Args:
        device (:class:`torch.device`): the desired device of the parameters
            and buffers in this module
        dtype (:class:`torch.dtype`): the desired floating point type of
            the floating point parameters and buffers in this module
        tensor (torch.Tensor): Tensor whose dtype and device are the desired
            dtype and device for all parameters and buffers in this module
        memory_format (:class:`torch.memory_format`): the desired memory
            format for 4D parameters and buffers in this module (keyword
            only argument)
    Returns:
        Module: self
    """
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
    if dtype is not None:
        if not dtype.is_floating_point:
            raise TypeError('nn.Module.to only accepts floating point '
                            'dtypes, but got desired dtype={}'.format(dtype))
    def convert(t):
        if convert_to_format is not None and t.dim() == 4:
            return t.to(device, dtype if t.is_floating_point() else None, non_blocking, memory_format=convert_to_format)
        return t.to(device, dtype if t.is_floating_point() else None, non_blocking)
    return self._apply(convert)
```

#### 1.1.8 state_dict()与load_state_dict()

​	state_dict()主要用于参数保存和重载，从state_dict()的源码可以看出它主要是一个递归过程，不断的进行子module的搜索。

```
def state_dict(self, destination=None, prefix='', keep_vars=False):
    r"""Returns a dictionary containing a whole state of the module.
    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.
    Returns:
        dict:
            a dictionary containing a whole state of the module
    """
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(version=self._version)
    self._save_to_state_dict(destination, prefix, keep_vars)
    # 通过递归获取所有子module的子module
    for name, module in self._modules.items():
        #　递归的结束条件就是当前module不包含子module
        if module is not None:
            module.state_dict(destination, prefix + name + '.', keep_vars=keep_vars)
    for hook in self._state_dict_hooks.values():
        hook_result = hook(self, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination
```

#### 1.1.9 parameters() & named_parameters()

​	named_parameters()函数的参数recurse用于确定是否递归，即是否遍历子module的子module，它同样是一个生成器函数。相同作用的函数还有buffers()和named_buffers()，用于返回buffer的两个generator；以及children()和named_children()，用于返回子module；还有modules()和named_modules()，用于递归返回所有子代module。

```
def named_parameters(self, prefix='', recurse=True):
    r"""Returns an iterator over module parameters, yielding both the
    name of the parameter as well as the parameter itself.
    Args:
        prefix (str): prefix to prepend to all parameter names.
        recurse (bool): if True, then yields parameters of this module
            and all submodules. Otherwise, yields only parameters that
            are direct members of this module.
    Yields:
        (string, Parameter): Tuple containing the name and parameter
    """
    gen = self._named_members(
        lambda module: module._parameters.items(),
        prefix=prefix, recurse=recurse)
    for elem in gen:
        yield elem

def parameters(self, recurse=True):
    r"""Returns an iterator over module parameters.
    This is typically passed to an optimizer.
    Args:
        recurse (bool): if True, then yields parameters of this module
            and all submodules. Otherwise, yields only parameters that
            are direct members of this module.
    Yields:
        Parameter: module parameter
    """
    # 遍历获得所有的parameters
    for name, param in self.named_parameters(recurse=recurse):
        yield param
```

#### 1.1.10 train() & eval()

​	训练模式和验证模式针对某些操作是不同的，比如“Dropout”和“BN”等，所以网络需要切换训练和测试模式，train()函数依旧是一个遍历过程，对每个子代module都进行设置。eval()模式仅仅需要将train()函数的mode设置为False即可:

```
def train(self, mode=True):
    r"""Sets the module in training mode.
    This has any effect only on certain modules. See documentations of
    particular modules for details of their behaviors in training/evaluation
    mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
    etc.
    Args:
        mode (bool): whether to set training mode (``True``) or evaluation
                     mode (``False``). Default: ``True``.
    Returns:
        Module: self
    """
    self.training = mode
    # 遍历
    for module in self.children():
        module.train(mode)
    return self

def eval(self):
    r"""
    Returns:
        Module: self
    """
    return self.train(False)
```

#### 1.1.11 requires_grad() & zero_grad()

​	requires_grad()控制自动求导是否记录求导结果，它是单个模块控制的

​	zeros_grad()的目的是针对所有的parameters进行统一的梯度清零操作，依旧是一个遍历过程:

```
def requires_grad_(self, requires_grad=True):
    for p in self.parameters():
        p.requires_grad_(requires_grad) #也是递归的调用，也是return self
    return self

def zero_grad(self):
    r"""Sets gradients of all model parameters to zero."""
    for p in self.parameters():
        # 判别此param是否含有grad
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
```















