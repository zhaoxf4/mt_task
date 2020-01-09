# coding=utf-8
# Neural Network for XOR
import numpy as np
import matplotlib.pyplot as plt

def change_bin(n):
    # 十进制转二进制的方法：除2取余，逆序排列
    # 输出结果前面至少有三个0，divide_array函数会用到 
    result = '000'
    if n == 0:    # 输入为0的情况
        return result
    else:
        result = change_bin(n // 2) # 调用自身
        return result + str(n % 2)
        
def divide_array(x, y):
    # 将输入的两个字符串整数按位分配到numpy数组的各个维度上
    # 利用np.array数组可以从右向左取的性质，所有字符串都能取到后三位
    x1 = int(x[-3])
    x2 = int(x[-2])
    x3 = int(x[-1])
    y1 = int(y[-3])
    y2 = int(y[-2])
    y3 = int(y[-1])
    result = np.array([[x1, y1], [x2, y2], [x3, y3]])
    return result

def change_int(X):
    # 将size为n * 1的numpy数组化为十进制整数
    result = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):     # 取整
            X[i][j] = int(round(X[i][j]))
        result += X[i][0] * (2 ** (X.shape[0]-1-i))
    return result

def rand_initialize_weights(L_in, L_out, epsilon):
    # 随机初始化一个权重矩阵W，维度L_out * 1+L_in，第一行是偏置
    epsilon_init = epsilon
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_gradient(z):
    g = np.multiply(sigmoid(z), (1 - sigmoid(z)))
    return g

def cost_function(theta1, theta2, X, y):
    m = X.shape[0]  # m=4
    # 计算所有参数的偏导数（梯度）
    D_1 = np.zeros(theta1.shape)  # Δ_1
    D_2 = np.zeros(theta2.shape)  # Δ_2
    h_total = np.zeros((m, 1))  # 所有样本的预测值, m*1
    for t in range(m):
        a_1 = np.vstack((np.array([[1]]), X[t:t + 1, :].T))  # 列向量, 3*1
        z_2 = np.dot(theta1, a_1)  # 2*1
        a_2 = np.vstack((np.array([[1]]), sigmoid(z_2)))  # 3*1
        z_3 = np.dot(theta2, a_2)  # 1*1
        a_3 = sigmoid(z_3)
        h = a_3  # 预测值h就等于a_3, 1*1
        h_total[t,0] = h
        delta_3 = h - y[t:t + 1, :].T  # 最后一层每一个单元的误差, 1*1
        delta_2 = np.multiply(np.dot(theta2[:, 1:].T, delta_3), sigmoid_gradient(z_2))  # 第二层每一个单元的误差（不包括偏置单元）,  2*1
        D_2 = D_2 + np.dot(delta_3, a_2.T)  # 第二层所有参数的误差, 1*3
        D_1 = D_1 + np.dot(delta_2, a_1.T)  # 第一层所有参数的误差, 2*3
    theta1_grad = (1.0 / m) * D_1  # 第一层参数的偏导数，取所有样本中参数的均值，没有加正则项
    theta2_grad = (1.0 / m) * D_2
    J = (1.0 / m) * np.sum(-y * np.log(h_total) - (np.array([[1]]) - y) * np.log(1 - h_total))
    return {'theta1_grad': theta1_grad,
            'theta2_grad': theta2_grad,
            'J': J, 'h': h_total}

def prediction(X):
    m = X.shape[0]  # m=3
    h_total = np.zeros((m, 1))  # 所有样本的预测值, m*1
    for t in range(m):
        a_1 = np.vstack((np.array([[1]]), X[t:t + 1, :].T))  # 列向量, 3*1
        z_2 = np.dot(theta1, a_1)  # 2*1
        a_2 = np.vstack((np.array([[1]]), sigmoid(z_2)))  # 3*1
        z_3 = np.dot(theta2, a_2)  # 1*1
        a_3 = sigmoid(z_3)
        h = a_3  # 预测值h就等于a_3, 1*1
        h_total[t,0] = h
    return h_total


if __name__ == '__main__':
    INPUT_LAYER = 2  # 输入层特征数
    HIDDEN_LAYER_SIZE = 2   # 隐藏层输入特征数
    NUM_LABELS = 1  # 输出层分类数
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    theta1 = rand_initialize_weights(INPUT_LAYER, HIDDEN_LAYER_SIZE, epsilon=1)  # epsilon不能太小
    theta2 = rand_initialize_weights(HIDDEN_LAYER_SIZE, NUM_LABELS, epsilon=1)
    
    iter_times = 10000  # 迭代次数
    alpha = 0.5  # 学习率
    result = {'J': [], 'h': []}
    theta_s = {}
    for i in range(iter_times):
        cost_fun_result = cost_function(theta1=theta1, theta2=theta2, X=X, y=y)
        theta1_g = cost_fun_result.get('theta1_grad')
        theta2_g = cost_fun_result.get('theta2_grad')
        J = cost_fun_result.get('J')
        h_current = cost_fun_result.get('h')
        theta1 -= alpha * theta1_g
        theta2 -= alpha * theta2_g
        result['J'].append(J)
        result['h'].append(h_current)
        # print(i, J, h_current)
        if i==0 or i==(iter_times-1):
            #print('theta1\n', theta1)
            #print('theta2\n', theta2)
            theta_s['theta1_'+str(i)] = theta1.copy()
            theta_s['theta2_'+str(i)] = theta2.copy()
    
    plt.plot(result.get('J'))
    plt.show()
    for key, value in theta_s.items():
        print(key+':\n', value)
    print('\n')
    print(result.get('h')[0], '\n\n', result.get('h')[-1])
    
    # 提示用户输入十进制数，由于input()的返回值是str类型，故需要转化为int类       
    num_a = int(input("请输入一个0-7的十进制数字："))
    num_b = int(input("请再次输入一个0-7的十进制数字："))
    X_input = divide_array(change_bin(num_a), change_bin(num_b))
    p = prediction(X_input)
    q = change_int(p)
    print(p, '\n', q)
    
