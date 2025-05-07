import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 假设你有多个 numpy 数组，每个数组代表一组数据
data1 = np.random.randn(100)  # 第一个数据集
data2 = np.random.randn(100)  # 第二个数据集

# 创建图形和多个子图
fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # 2行1列的子图，纵向排列

# 为每个子图设置范围
axs[0].set_xlim(0, 50)
axs[0].set_ylim(np.min(data1)-1, np.max(data1)+1)
axs[1].set_xlim(0, 50)
axs[1].set_ylim(np.min(data2)-1, np.max(data2)+1)

# 创建两个空的线对象，用于更新数据
line1, = axs[0].plot([], [], lw=2)
line2, = axs[1].plot([], [], lw=2)

# 开启交互模式
plt.ion()

# 更新函数，动态更新图形
def update(frame):
    global data1, data2
    # 模拟每次程序运行时添加新的数据
    data1 = np.append(data1, np.random.randn())  # 更新第一个数据集
    data2 = np.append(data2, np.random.randn())  # 更新第二个数据集

    # 保留最新的50个数据点
    if len(data1) > 50:
        data1 = data1[-50:]
    if len(data2) > 50:
        data2 = data2[-50:]

    # 更新线条数据
    line1.set_data(range(len(data1)), data1)
    line2.set_data(range(len(data2)), data2)

    # 刷新图形
    plt.draw()
    plt.pause(0.04)  # 这里可以调整刷新频率

# 模拟其他代码的运行
import time

# 假设数据每隔0.5秒就更新一次
for _ in range(100):  # 模拟100次更新
    update(_)  # 更新图形
    time.sleep(0.001)  # 模拟其他代码的运行

# 关闭交互模式，保持图形窗口
plt.ioff()
plt.show()
