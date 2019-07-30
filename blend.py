import cv2
import numpy as np





def huandiao(A,B,occ):  
    #################################################
#    A = result.copy()
#    B = ref2src.copy()
#    A=cv2.resize(A,(640,960)).astype(np.float32)
#    B=cv2.resize(B,(640,960)).astype(np.float32)
    A=A.astype(np.float32)
    B=B.astype(np.float32)
    print(A.dtype)
    occ=np.float32(occ)
#    kernel=np.ones([21,21])
#    occ=cv2.dilate(occ,kernel)
#    occ=cv2.resize(occ,(640,960),interpolation=cv2.INTER_NEAREST)
    row,col,_=A.shape
    col_list=[col,col//2,col//4,col//8,col//16,col//32]
    row_list=[row,row//2,row//4,row//8,row//16,row//32]


    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(5):
#        G = cv2.pyrDown(G)
        row,col,_=G.shape
        G=cv2.resize(G,(col_list[i+1],row_list[i+1]))
        gpA.append(G)
    
    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(5):
#        G = cv2.pyrDown(G)
        row,col,_=G.shape
        G=cv2.resize(G,(col_list[i+1],row_list[i+1]))
        gpB.append(G)
    
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5,0,-1):
#        GE = cv2.pyrUp(gpA[i])
        
#        L = cv2.subtract(gpA[i-1],GE)
        row,col,_=gpA[i].shape
        GE=cv2.resize(gpA[i],(col_list[i-1],row_list[i-1])) 
        L=gpA[i-1]-GE
        lpA.append(L)
    
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5,0,-1):
#        GE = cv2.pyrUp(gpB[i])
#        L = cv2.subtract(gpB[i-1],GE)
        row,col,_=gpB[i].shape

        GE=cv2.resize(gpB[i],(col_list[i-1],row_list[i-1]))
        L=gpB[i-1]-GE
        lpB.append(L)
    
    # Now add left and right halves of images in each level
    LS = []
    for la,lb in zip(lpA,lpB):
        rows,cols,dpt = la.shape
        print(la.shape)
#        ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
        occ_temp=cv2.resize(occ,(cols,rows))#,interpolation=cv2.INTER_NEAREST)

        occ_temp_3=np.zeros_like(la)
        occ_temp_3[:,:,0]=occ_temp.copy()
        occ_temp_3[:,:,1]=occ_temp.copy()
        occ_temp_3[:,:,2]=occ_temp.copy()
        ls=la*(1-occ_temp_3)+lb*occ_temp_3
        LS.append(ls)
    
    # now reconstruct
    ls_ = LS[0]
    for i in range(1,6):
        row,col,_=ls_.shape
#        ls_ = cv2.resize(ls_,(col*2,row*2))#cv2.pyrUp(ls_)
        ls_ = cv2.resize(ls_,(col_list[5-i],row_list[5-i]))#cv2.pyrUp(ls_)
        ls_ = ls_+LS[i]#cv2.add(ls_, LS[i])

# image with direct connecting each half

    return np.clip(ls_,0,255).astype(np.uint8)
