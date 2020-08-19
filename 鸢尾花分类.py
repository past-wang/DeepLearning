#准备数据
    #数据集读入
    #数据集乱序
    #生成训练集和测试集
    #配成（输入特征，标签）对,每次读入一小撮(batch)
#搭建网络
#参数优化
#测试效果
#acc/loss可视化

#读入数据集
from sklearn import datasets
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

x_data = datasets.load_iris().data#输入特征
y_data = datasets.load_iris().target#输入标签

#数据集乱序,信息应该是杂乱无章的，让数据集乱序，使用同样的随机种子
np.random.seed(116)#使用相同的seed，使得输入特征/标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

#数据集分出永不相见的训练集和测试集
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

#转换x的数据类型，否则后面可能数据类型不一致
x_train = tf.cast(x_train,tf.float32)
x_test = tf.cast(x_test,tf.float32)

#配成[输入特征，标签]对，每次喂入一小撮（batch）
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

#定义神经网络中所有可训练参数,y=wx+b
w1 = tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=1))#4个输入，三个输出
b1 = tf.Variable(tf.random.truncated_normal([3],stddev=0.1,seed=1))#输出结点为3种


lr=0.1
train_loss_results = []#定义学习率参数
test_acc = []#将每轮的acc记录在这个列表中
epoch =5000#循环次数
loss_all = 0#每轮分4个step.loss_all记录四个step生成的4个loss和

# 训练部分
for epoch in range(epoch):  #数据集级别的循环，每个epoch循环一次数据集
    for step, (x_train, y_train) in enumerate(train_db):  #batch级别的循环 ，每个step循环一个batch
        with tf.GradientTape() as tape:  # with结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算----> y=w*x+b
            y = tf.nn.softmax(y)  # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）--->作为一个概率输出
            y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热码格式，方便计算loss和accuracy
            loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_all += loss.numpy()  # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])#------>求导的操作

        # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
        w1.assign_sub(lr * grads[0])  # 参数w1自更新
        b1.assign_sub(lr * grads[1])  # 参数b自更新

    # 每个epoch，打印loss信息
    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
    loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备

    # 测试部分
    # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1#——————>y=w1*x_test+b1
        y = tf.nn.softmax(y)#----->类似于归一化
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        # 将pred转换为y_test的数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)#——————>强制类型转换为tf.int32
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
        total_number += x_test.shape[0]
    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("--------------------------")
    
#绘制loss曲线
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results,label="$Loss$")
plt.legend()
plt.show()

plt.title('Acc curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc,label="$Accuracy$")
plt.legend()
plt.show()