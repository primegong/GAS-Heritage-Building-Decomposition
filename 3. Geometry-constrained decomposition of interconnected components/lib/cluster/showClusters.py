import matplotlib.pyplot as plt

def show_features(features):
    plt.figure(figsize=(10,6))
    plt.plot(features[:, 0], features[:,1], 'o', markerfacecolor=(0,0,255),markeredgecolor='k', markersize=14)
    plt.show()


def show_clusters(labels, X):

    core_samples_mask = np.zeros_like(labels, dtype=bool)  # 设置一个样本个数长度的全false向量
    core_samples_mask[db.core_sample_indices_] = True #将核心样本部分设置为true
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure(figsize=(10,6))
    for k, col in zip(unique_labels, colors):
        if k == -1:  # 聚类结果为-1的样本为离散点
            # 使用黑色绘制离散点
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)  # 将所有属于该聚类的样本位置置为true

        xy = X[class_member_mask & core_samples_mask]  # 将所有属于该类的核心样本取出，使用大图标绘制
        
        plt.plot(xy[:, 0], [0]*len(xy), 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]  # 将所有属于该类的非核心样本取出，使用小图标绘制
        plt.plot(xy[:, 0], [0]*len(xy), 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
