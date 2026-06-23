

import numpy as np
from sklearn.cluster import DBSCAN

def dbSCAN(features, show = False):
    features = np.array(features).astype(np.float64)

    ### 基于点密度的DBSACN聚类: 二维点密度
    db = DBSCAN(eps=0.3, min_samples=10)
    db.fit(features)
    # eps：ϵ参数，用于确定邻域大小。
    # min_samples：用于判断核心对象。
    ## 关键点：若满足在eps半径内有至少min_samples个点，则为关键点，可以继续搜索；否则为叶子点，停止搜索，集群完成。
    labels = db.labels_
    n_clusters_ = len(np.unique(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    
    # core_samples_mask = np.zeros_like(labels, dtype = bool)
    # core_samples_mask[db.core_sample_indices_] = True


    '''
   这里是关键点(针对这行代码：xy = X[class_member_mask & ~core_samples_mask])：
   db.core_sample_indices_  表示的是某个点在寻找核心点集合的过程中暂时被标为噪声点的点(即周围点
   小于min_samples)，并不是最终的噪声点。在对核心点进行联通的过程中，这部分点会被进行重新归类(即标签
   并不会是表示噪声点的-1)，也可也这样理解，这些点不适合做核心点，但是会被包含在某个核心点的范围之内
    '''
    print('num_clusters:', len(unique_labels))
    for k in unique_labels:
      ##-1表示噪声点,这里的k表示黑色
      ##生成一个True、False数组，lables == k 的设置成True
      class_member_mask = (labels == k)
      index = np.where(class_member_mask==True)[0]


      # ##两个数组做&运算，找出即是核心点又等于分类k的值  markeredgecolor='k',
      # index = np.where(class_member_mask & core_samples_mask==True)[0]
      clusters = []
      for k in unique_labels:
          ##-1表示噪声点,这里的k表示黑色
          ##生成一个True、False数组，lables == k 的设置成True
          class_member_mask = (labels == k)
          index = np.where(class_member_mask==True)[0]
          clusters.append(index)

    ##### sort ############
    clusters = sorted(clusters, key = lambda x:len(x), reverse=True)

    return clusters




