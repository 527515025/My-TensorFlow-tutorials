
import matplotlib.pyplot as plt
from PIL import Image

def get_bin_table(threshold=140):
    """
    获取灰度转二值的映射table
    :param threshold:
    :return:
    """
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
 
    return table


image = Image.open('img/0UnZ.jpg')

imgry = image.convert('L')# 转化为灰度图
table = get_bin_table()
out = imgry.point(table, '1')
print(out)

f = plt.figure()
plt.imshow(out)
plt.show()



