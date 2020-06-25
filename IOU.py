class BBox:
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
def iou(a,b):
    #assert语句是一种插入调试断点到程序的一种便捷的方式。
    assert isinstance(a,BBox)
    assert isinstance(b,BBox)
    area_a = a.w * a.h
    area_b = b.w * b.h
    w = min(a.x+a.w,b.x+b.w) - max(a.x,b.x)
    h = min(a.y+a.h,b.y+b.h) - max(b.y,a.y)
    #处理特殊情况,当w或者h小于0的情况出现，返回零
    if w <= 0 or h <= 0:
        return 0
    area_c = w * h
    return area_c / ( area_a + area_b - area_c)

if __name__=='__main__':
    a = BBox(1,1,4,5)
    b1 = BBox(1,1,4,5)
    b2 = BBox(5,1,4,5)
    b3 = BBox(3,2,3,6)

    print("iou:", iou(a, b1))
    print("iou:", iou(a, b2))
    print("iou:", iou(a, b3))
