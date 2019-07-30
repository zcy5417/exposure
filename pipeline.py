import cv2
import utils

from imp import reload 
reload(utils)



## main
import time   
low=cv2.imread('1.jpg')
ref=cv2.imread('2.jpg')
high=cv2.imread('3.jpg')
#hhigh=cv2.imread('4.jpg')
begin=time.time()
res_mertens_8bit=utils.hdr(low,ref,high)
end=time.time()
print('run time',end-begin)
cv2.imwrite("fusion_mertens_ww7.jpg", res_mertens_8bit)



