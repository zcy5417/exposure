import cv2
import blend
from imp import reload 
reload(blend)
import numpy as np
#import matplotlib.pyplot as plt

def his_match(src, dst):
    res = np.zeros_like(dst)
    # cdf 为累计分布
    cdf_src = np.zeros((3, 256))
    cdf_dst = np.zeros((3, 256))
    cdf_res = np.zeros((3, 256))
    kw = dict(bins=256, range=(0, 256), density=True)
    for ch in range(3):
        his_src, _ = np.histogram(src[:, :, ch], **kw)
        hist_dst, _ = np.histogram(dst[:, :, ch], **kw)
        
        
        cdf_src[ch] = np.cumsum(his_src)
        cdf_dst[ch] = np.cumsum(hist_dst)
        index = np.searchsorted(cdf_src[ch], cdf_dst[ch], side='left')
        np.clip(index, 0, 255, out=index)
        res[:, :, ch] = index[dst[:, :, ch]]
        his_res, _ = np.histogram(res[:, :, ch], **kw)
        cdf_res[ch] = np.cumsum(his_res)
    return res, (cdf_src, cdf_dst, cdf_res)



#def his_match_clip(src, dst):
#    res = np.zeros_like(dst)
#    # cdf 为累计分布
#    cdf_src = np.zeros((3, 256))
#    cdf_dst = np.zeros((3, 256))
#    cdf_res = np.zeros((3, 256))
#    kw = dict(bins=256, range=(0, 256), density=True)
#    
##    dst_temp=dst.copy()
#    dst_gray=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
#    mask=(dst_gray<0.98*255)&(dst_gray>0.02*255)
#    
#    
#    for ch in range(3):
#        his_src, _ = np.histogram(src[mask, ch], **kw)
#        hist_dst, _ = np.histogram(dst[mask, ch], **kw)
#        
#        
#        
#        cdf_src[ch] = np.cumsum(his_src)
#        cdf_dst[ch] = np.cumsum(hist_dst)
#        index = np.searchsorted(cdf_src[ch], cdf_dst[ch], side='left')
#        np.clip(index, 0, 255, out=index)
#        res[:, :, ch] = index[dst[:, :, ch]]
#        res[~mask,ch]=dst[~mask,ch]
#        
#        his_res, _ = np.histogram(res[:, :, ch], **kw)
#        cdf_res[ch] = np.cumsum(his_res)
#    return res, (cdf_src, cdf_dst, cdf_res)



#def his_match_tihuan(src, dst):
#    res = np.zeros_like(dst)
#    # cdf 为累计分布
#    cdf_src = np.zeros((3, 256))
#    cdf_dst = np.zeros((3, 256))
#    cdf_res = np.zeros((3, 256))
#    kw = dict(bins=256, range=(0, 256), density=True)
#    for ch in range(3):
#        his_src, _ = np.histogram(src[:, :, ch], **kw)
#        hist_dst, _ = np.histogram(dst[:, :, ch], **kw)
#        
#        ###???
#        temp=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
##        mask=temp>240
#        mask=temp>255###################
#        
#        
#        
#        cdf_src[ch] = np.cumsum(his_src)
#        cdf_dst[ch] = np.cumsum(hist_dst)
#        index = np.searchsorted(cdf_src[ch], cdf_dst[ch], side='left')
#        np.clip(index, 0, 255, out=index)
#        res[:, :, ch] = index[dst[:, :, ch]]
#        his_res, _ = np.histogram(res[:, :, ch], **kw)
#        cdf_res[ch] = np.cumsum(his_res)
#        
#        
#        
#        res[mask,:]=src[mask,:]
##        res=blend.huandiao(res,src,mask)
#    return res, (cdf_src, cdf_dst, cdf_res)
########### histmatch
    

def warp(src,ref):
#    ref = cv2.imread('2.jpg')
#    src = cv2.imread('3.jpg')
#    
    ref2src, cdfs = his_match(src, ref)
#    ref2src,_=his_match_clip(src,ref)
    
#    ref2src,_=his_match_tihuan(src,ref)
    
#    ref2src=ref.copy()
    #cv.imshow('res', res)
#    cv2.imwrite('ref2high.jpg',res)

    a=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    b=cv2.cvtColor(ref2src,cv2.COLOR_BGR2GRAY)
    inst = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOpticalFlow_PRESET_MEDIUM)
    inst.setUseSpatialPropagation(True)
    flow1 = inst.calc(a, b, None)
    flow2 = inst.calc(b, a, None)
    
    
    row,col,_=flow1.shape
    y,x=np.meshgrid(np.arange(0,col),np.arange(0,row))
    yy=y+flow2[:,:,0]
    xx=x+flow2[:,:,1]
    yy=yy.astype(np.float32)
    xx=xx.astype(np.float32)
    flow1_to_flow2 = cv2.remap(flow1,yy,xx,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    
    

#    diff=flow2+flow1_to_flow2
#    diff_sum=(np.sum(diff**2,axis=2))**0.5
#    zz=diff_sum>2#1/320*row
    diff=flow2+flow1_to_flow2
    lhs=np.sum(diff**2,axis=2)
    rhs=0.01*(np.sum(flow2**2,axis=2)+np.sum(flow1_to_flow2**2,axis=2))+0.5
    zz=lhs>=rhs
#    plt.figure()
#    plt.imshow(zz)
    
    row,col=a.shape
    y,x=np.meshgrid(np.arange(0,col),np.arange(0,row))
    
    ref_gray=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
    cliped_region=(ref_gray<0.02*255)|(ref_gray>0.98*255)
    #flow2[cliped_region,:]=interpolate_flow2[cliped_region,:]
    
    new_cliped=zz&cliped_region
    new_zz=zz^new_cliped
#    plt.figure(),plt.imshow(new_cliped)
#    plt.figure(),plt.imshow(new_zz)
    
    
#    ref_gray=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)

    
    
    
    
    
    
    flow2[new_cliped,:]=np.mean(np.mean(flow2,axis=0),axis=0)
    yy=y+flow2[:,:,0]
    xx=x+flow2[:,:,1]
    yy=yy.astype(np.float32)
    xx=xx.astype(np.float32)
    
    
    result=cv2.remap(src,yy,xx,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
#    result[new_zz,:]=ref2src[new_zz,:]
    result=blend.huandiao(result,ref2src,new_zz)
#    cv2.imwrite('high_warp_clip.jpg',result)
    return result





def hdr(low,ref,high):
    low_warp=warp(low,ref)
    high_warp=warp(high,ref)
#    hhigh_warp=warp(hhigh,ref)
    #cv2.imwrite('high_warp_qq.jpg',high_warp)
    #cv2.imwrite('low_warp_qq.jpg',low_warp)

    
    img_list = [low_warp,ref,high_warp]
    
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(img_list)

    
    res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
    return res_mertens_8bit









