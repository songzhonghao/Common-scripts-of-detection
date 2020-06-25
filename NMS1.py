'''
目标检测中常用到NMS，在faster R-CNN中，每一个bounding box都有一个打分，NMS实现逻辑是：
    1，按打分最高到最低将BBox排序 ，例如：A B C D E F
    2，A的分数最高，保留。从B-E与A分别求重叠率IoU，假设B、D与A的IoU大于阈值，那么B和D可以认为是重复标记去除
    3，余下C E F，重复前面两步。
'''
import numpy as np
# 不区分类别的nms
def nms(dets, thresh):
    """Pure Python NMS baseline."""
    # 取出所有预测框的x1,y1,x2,y2
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    # 计算每一个框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 将置信度得分按照逆序排列，打分从大到小排列，取index
    order = scores.argsort()[::-1]
    # # keep为最后保留的边框，keep是保留的预测框的索引
    keep = []
    while order.size > 0:
        i = order[0]
        # order经过排序后得到，因此第一个是得分最高的
        # order保存的是所有预测框经过排序后的索引，索引还是排序前的索引，只不过经过了排序
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        over = inter / (areas[i] + areas[order[1:]] - inter)
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        #inds是所有预测框与窗口i的iou小于thresh的窗口的索引，这里的索引没有加入窗口i，因此inds的索引对应到order中应该是少1位
        inds = np.where(over <= thresh)[0]
        order = order[inds + 1]
    return dets[keep]

def iou(a,b):
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    w = min(a[2],b[2]) - max(a[0],b[0])
    h = min(a[3],b[3]) - max(b[1],a[1])
    #处理特殊情况,当w或者h小于0的情况出现，返回零
    if w <= 0 or h <= 0:
        return 0
    area_c = w * h
    return area_c / ( area_a + area_b - area_c)

#采用不排序的方法进行nms
# 1、每次选取一个预测框，依次计算与其他框的iou
# 2、如果两个框的iou大于设定值thresh，则比较两个预测框的置信度得分，将得分低的那个预测框置0
# 3，删除预测框为0的预测框
# 时间复杂度O(n^2)
def nms2(dets,thresh):
    for i in range(len(dets)-1):
        for j in range(i+1,len(dets)):
            if iou(dets[i],dets[j]) > thresh:
                if dets[i][-1] > dets[j][-1]:
                    dets[j] = [0, 0, 0, 0, 0]
                else:
                    dets[i] = [0, 0, 0, 0, 0]
            else:
                continue
    for det in dets:
        if det == [0, 0, 0, 0, 0]:
            dets.remove(det)
    return dets

if __name__=='__main__':
    print(nms2([[661, 27, 679, 47, 0.95],
                [662, 27, 682, 47, 0.90],
                [663, 26, 685, 48, 0.55],
                [662, 27, 682, 47, 0.75]],
                0.85))