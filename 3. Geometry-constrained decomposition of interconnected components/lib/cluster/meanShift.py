
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth


def meanShift(features, quantile_value = 0.1):
    '''
    '''
 
    features = np.array(features).astype(np.float64)

    '''本质上就是求平均最远k近邻距离, quantile的值表示进行近邻搜索时候的近邻占样本的比例'''
    bandwidth = estimate_bandwidth(features, quantile=quantile_value)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(features)
    labels = ms.labels_
    unique_labels = set(labels)

    # core_samples_mask = np.zeros_like(labels, dtype = bool)
    # core_samples_mask[ms.core_sample_indices_] = True

    clusters = []
    for k in unique_labels:
        ##-1表示噪声点,这里的k表示黑色
        ##生成一个True、False数组，lables == k 的设置成True
        class_member_mask = (labels == k)
        index = np.where(class_member_mask==True)[0]        
        ##两个数组做&运算，找出即是核心点又等于分类k的值  markeredgecolor='k',
        # index = np.where(class_member_mask & core_samples_mask==True)[0]

        clusters.append(index)

    ##### sort ############
    clusters = sorted(clusters, key = lambda x:len(x), reverse=True)
    
    return clusters

