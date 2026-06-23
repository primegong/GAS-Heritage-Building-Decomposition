import cv2
 
# 输入需要细化的图片（经过二值化处理的图片）和映射矩阵array
def Thin(image, array):
    h, w = image.shape
    # height 图像的高度 weight 图像的宽度
    iThin = image
    point_list = []
    for i in range(h):
        for j in range(w):
            if image[i, j] == 0:  # 如果图像的这个点是黑色的点
                a = [1] * 9  # 先默认周围八个点都是白色的
                for k in range(3):   # range(3):0 1 2
                    for l in range(3):
                        # 如果3*3矩阵的点不在边界
                        # 且是黑色的点
                        # 这一段的难点是用k，l来精确的表示3*3的矩阵
                        if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and iThin[i - 1 + k, j - 1 + l] == 0:
                            a[k + 3 * l] = 0
                # sum没有a[4]，a[4]刚好就是中心的黑点，不需要加进去计算
                sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                # 然后根据array表，对iThin的那一点进行赋值。
                # 将所有点的总价值映射到[0,255]的表中，返回的结果保存到iThin中
                point_list.append([i,j])
                iThin[i, j] = array[sum] * 255
    
    return iThin, point_list
 
 
# 最简单的二值化函数，阈值根据图片的昏暗程度自己设定，即180这个数字是可以修改的
def Two(image):
    w, h = image.shape
    # size = (w, h)
    iTwo = image
    for i in range(w):
        for j in range(h):
            iTwo[i,j] = 0 if image[i,j] < 180 else 255
    return iTwo
 
def skeleton(imgfile):
    # 映射表
    array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
 
 
    # 读取图片，并显示
    img = cv2.imread(imgfile, 0)
 
    # convert image first 
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    converted_img = 255-img
 
    # # 获取简单二值化的细化图，并显示
    # iTwo = Two(converted_img)

    iThin_2, point_list = Thin(converted_img, array)
    iThin_2 = 255-iThin_2
    # iThin_2 = cv2.GaussianBlur(iThin_2, (3, 3), 0)
    return iThin_2

if __name__ == '__main__':
    input_img('Tile_+026_+028_1_X-Y_bird.png')