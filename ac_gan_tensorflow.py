
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np  # 定义矩阵
import matplotlib.pyplot as plt  # 画图
import matplotlib.gridspec as gridspec  # 定义网格
import os

mnist = input_data.read_data_sets('../MNIST', one_hot=True)

mb_size = 32
X_dim = mnist.train.images.shape[1]  # 图像维度 [数据量，图像维度]
y_dim = mnist.train.labels.shape[1]  # Label 数据长
z_dim = 10  # 噪音维度
h_dim = 128  # 中间层神经元数
eps = 1e-8  # 定义一个很小的数 +eps 可保证不为0
lr = 1e-3  # 学习率
d_steps = 3  # 没用到


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')  # cmp:

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


def generator(z, c):
    inputs = tf.concat(axis=1, values=[z, c])  # ACGAN 把噪音和标签concat作为G的输入
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2  # logit值
    G_prob = tf.nn.sigmoid(G_log_prob)  # 返回图像G_prob
    return G_prob


D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_W2_gan = tf.Variable(xavier_init([h_dim, 1]))  # 判断0/1的GAN的Weight
D_b2_gan = tf.Variable(tf.zeros(shape=[1]))  # 判断0/1的GAN的Bias
D_W2_aux = tf.Variable(xavier_init([h_dim, y_dim]))  # 判断类别 Weight
D_b2_aux = tf.Variable(tf.zeros(shape=[y_dim]))  # 判断类别 Bias


def discriminator(X):
    D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    out_gan = tf.nn.sigmoid(tf.matmul(D_h1, D_W2_gan) + D_b2_gan)  # 输出fake/real 的概率值
    out_aux = tf.matmul(D_h1, D_W2_aux) + D_b2_aux  # 输出class 的概率值
    return out_gan, out_aux


theta_G = [G_W1, G_W2, G_b1, G_b2]
theta_D = [D_W1, D_W2_gan, D_W2_aux, D_b1, D_b2_gan, D_b2_aux]


def sample_z(m, n):  # 产生随机噪音
    return np.random.uniform(-1., 1., size=[m, n])


def cross_entropy(logit, y):  #
    return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))


G_sample = generator(z, y)  # G产生的fake image

D_real, C_real = discriminator(X)
D_fake, C_fake = discriminator(G_sample)

# Cross entropy aux loss  # 标签误差
C_loss = cross_entropy(C_real, y) + cross_entropy(C_fake, y)

# GAN D loss
D_loss = tf.reduce_mean(tf.log(D_real + eps) + tf.log(1. - D_fake + eps))  # +eps 加上一个很小的数 防止为0
DC_loss = -(D_loss + C_loss)

# GAN's G loss
G_loss = tf.reduce_mean(tf.log(D_fake + eps))
GC_loss = -(G_loss + C_loss)

D_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(DC_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(GC_loss, var_list=theta_G))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):  # 输入G产生的图片的文件夹
    os.makedirs('out/')

i = 0

for it in range(1000000):
    X_mb, y_mb = mnist.train.next_batch(mb_size)
    z_mb = sample_z(mb_size, z_dim)  # 产生batch的噪音z

    _, DC_loss_curr = sess.run(
        [D_solver, DC_loss],
        feed_dict={X: X_mb, y: y_mb, z: z_mb}
    )

    _, GC_loss_curr = sess.run(
        [G_solver, GC_loss],
        feed_dict={X: X_mb, y: y_mb, z: z_mb}
    )

    if it % 1000 == 0:
        idx = np.random.randint(0, 10)  # MNIST数据一共10类
        c = np.zeros([16, y_dim])
        c[range(16), idx] = 1  # 产生16个随机one-hot行向量

        samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim), y: c})  # 用 it 次的G产生16张图像并plot

        print('Iter: {}; DC_loss: {:.4}; GC_loss: {:.4}; Idx： {}'
              .format(it, DC_loss_curr, GC_loss_curr, idx))

        fig = plot(samples)
        plt.savefig('out/{}.png'
                    .format(str(idx)+'_'+str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
