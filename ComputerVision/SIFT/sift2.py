import warnings
import cv2
warnings.filterwarnings("ignore")
import numpy as np


def GuassianKernel(sigma , dim):
    temp = [t - (dim/2) for t in range(dim)]
    assistant = []
    for i in range(dim):
        assistant.append(temp)
    assistant = np.array(assistant)
    temp = 2*sigma*sigma
    result = (1.0/(temp*np.pi))*np.exp(-(assistant**2+(assistant.T)**2)/temp)
    return result

def getDoG(img,n,sigma0,S = None,O = None):

    if S == None:
        S = n + 3
    if O == None:
        O = int(np.log2(min(img.shape[0], img.shape[1]))) - 3

    k = 2 ** (1.0 / n)
    sigma = [[(k**s)*sigma0*(1<<o) for s in range(S)] for o in range(O)]
    samplePyramid = [downsample(img, 1 << o) for o in range(O)]

    GuassianPyramid = []
    for i in range(O):
        GuassianPyramid.append([])
        for j in range(S):
            dim = int(6*sigma[i][j] + 1)
            if dim % 2 == 0:
                dim += 1
            GuassianPyramid[-1].append(convlove(GuassianKernel(sigma[i][j], dim),samplePyramid[i],[dim/2,dim/2,dim/2,dim/2],[1,1]))
    DoG = [[GuassianPyramid[o][s+1] - GuassianPyramid[o][s] for s in range(S - 1)] for o in range(O)]


    return DoG,GuassianPyramid

def getDoG(img,n,sigma0,S = None,O = None):

    if S == None:
        S = n + 3
    if O == None:
        O = int(np.log2(min(img.shape[0], img.shape[1]))) - 3

    k = 2 ** (1.0 / n)
    sigma = [[(k**s)*sigma0*(1<<o) for s in range(S)] for o in range(O)]
    samplePyramid = [downsample(img, 1 << o) for o in range(O)]

    GuassianPyramid = []
    for i in range(O):
        GuassianPyramid.append([])
        for j in range(S):
            dim = int(6*sigma[i][j] + 1)
            if dim % 2 == 0:
                dim += 1
            GuassianPyramid[-1].append(convlove(GuassianKernel(sigma[i][j], dim),samplePyramid[i],[dim/2,dim/2,dim/2,dim/2],[1,1]))
    DoG = [[GuassianPyramid[o][s+1] - GuassianPyramid[o][s] for s in range(S - 1)] for o in range(O)]


    return DoG,GuassianPyramid



def convlove(filter,mat,padding,strides):
    result = None
    filter_size = filter.shape
    mat_size = mat.shape
    if len(filter_size) == 2:
        if len(mat_size) == 3:
            channel = []
            for i in range(mat_size[-1]):
                pad_mat = np.pad(mat[:,:,i], ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
                temp = []
                for j in range(0,mat_size[0],strides[1]):
                    temp.append([])
                    for k in range(0,mat_size[1],strides[0]):
                        val = (filter*pad_mat[j*strides[1]:j*strides[1]+filter_size[0],
                                      k*strides[0]:k*strides[0]+filter_size[1]]).sum()
                        temp[-1].append(val)
                channel.append(np.array(temp))

            channel = tuple(channel)
            result = np.dstack(channel)
        elif len(mat_size) == 2:
            channel = []
            padding[0] = int(padding[0])
            padding[1] = int(padding[1])
            padding[2] = int(padding[2])
            padding[3] = int(padding[3])


            pad_mat = np.pad(mat, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
            for j in range(0, mat_size[0], strides[1]):
                channel.append([])
                for k in range(0, mat_size[1], strides[0]):
                    val = (filter * pad_mat[j * strides[1]:j * strides[1] + filter_size[0],
                                    k * strides[0]:k * strides[0] + filter_size[1]]).sum()
                    channel[-1].append(val)


            result = np.array(channel)

    return result

def adjustLocalExtrema(DoG,o,s,x,y,contrastThreshold,edgeThreshold,sigma,n,SIFT_FIXPT_SCALE):
    SIFT_MAX_INTERP_STEPS = 5
    SIFT_IMG_BORDER = 5

    point = []

    img_scale = 1.0 / (255 * SIFT_FIXPT_SCALE)
    deriv_scale = img_scale * 0.5
    second_deriv_scale = img_scale
    cross_deriv_scale = img_scale * 0.25

    img = DoG[o][s]
    i = 0
    while i < SIFT_MAX_INTERP_STEPS:
        if s < 1 or s > n or y < SIFT_IMG_BORDER or y >= img.shape[1] - SIFT_IMG_BORDER or x < SIFT_IMG_BORDER or x >= img.shape[0] - SIFT_IMG_BORDER:
            return None,None,None,None

        img = DoG[o][s]
        prev = DoG[o][s - 1]
        next = DoG[o][s + 1]

        dD = [ (img[x,y + 1] - img[x, y - 1]) * deriv_scale,
               (img[x + 1, y] - img[x - 1, y]) * deriv_scale,
               (next[x, y] - prev[x, y]) * deriv_scale ]

        v2 = img[x, y] * 2
        dxx = (img[x, y + 1] + img[x, y - 1] - v2) * second_deriv_scale
        dyy = (img[x + 1, y] + img[x - 1, y] - v2) * second_deriv_scale
        dss = (next[x, y] + prev[x, y] - v2) * second_deriv_scale
        dxy = (img[x + 1, y + 1] - img[x + 1, y - 1] - img[x - 1, y + 1] + img[x - 1, y - 1]) * cross_deriv_scale
        dxs = (next[x, y + 1] - next[x, y - 1] - prev[x, y + 1] + prev[x, y - 1]) * cross_deriv_scale
        dys = (next[x + 1, y] - next[x - 1, y] - prev[x + 1, y] + prev[x - 1, y]) * cross_deriv_scale

        H=[ [dxx, dxy, dxs],
            [dxy, dyy, dys],
            [dxs, dys, dss]]

        X = np.matmul(np.linalg.pinv(np.array(H)),np.array(dD))

        xi = -X[2]
        xr = -X[1]
        xc = -X[0]

        if np.abs(xi) < 0.5 and np.abs(xr) < 0.5 and np.abs(xc) < 0.5:
            break

        y += int(np.round(xc))
        x += int(np.round(xr))
        s += int(np.round(xi))

        i+=1

    if i >= SIFT_MAX_INTERP_STEPS:
        return None,x,y,s
    if s < 1 or s > n or y < SIFT_IMG_BORDER or y >= img.shape[1] - SIFT_IMG_BORDER or x < SIFT_IMG_BORDER or x >= \
            img.shape[0] - SIFT_IMG_BORDER:
        return None, None, None, None


    t = (np.array(dD)).dot(np.array([xc, xr, xi]))

    contr = img[x,y] * img_scale + t * 0.5
    if np.abs( contr) * n < contrastThreshold:
        return None,x,y,s


    # 利用Hessian矩阵的迹和行列式计算主曲率的比值
    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy
    if det <= 0 or tr * tr * edgeThreshold >= (edgeThreshold + 1) * (edgeThreshold + 1) * det:
        return None,x,y,s

    point.append((x + xr) * (1 << o))
    point.append((y + xc) * (1 << o))
    point.append(o + (s << 8) + (int(np.round((xi + 0.5)) * 255) << 16))
    point.append(sigma * np.power(2.0, (s + xi) / n)*(1 << o) * 2)

    return point,x,y,s

def GetMainDirection(img,r,c,radius,sigma,BinNum):
    expf_scale = -1.0 / (2.0 * sigma * sigma)

    X = []
    Y = []
    W = []
    temphist = []

    for i in range(BinNum):
        temphist.append(0.0)

    # 图像梯度直方图统计的像素范围
    k = 0
    for i in range(-radius,radius+1):
        y = r + i
        if y <= 0 or y >= img.shape[0] - 1:
            continue
        for j in range(-radius,radius+1):
            x = c + j
            if x <= 0 or x >= img.shape[1] - 1:
                continue

            dx = (img[y, x + 1] - img[y, x - 1])
            dy = (img[y - 1, x] - img[y + 1, x])

            X.append(dx)
            Y.append(dy)
            W.append((i * i + j * j) * expf_scale)
            k += 1


    length = k

    W = np.exp(np.array(W))
    Y = np.array(Y)
    X = np.array(X)
    Ori = np.arctan2(Y,X)*180/np.pi
    Mag = (X**2+Y**2)**0.5

    # 计算直方图的每个bin
    for k in range(length):
        bin = int(np.round((BinNum / 360.0) * Ori[k]))
        if bin >= BinNum:
            bin -= BinNum
        if bin < 0:
            bin += BinNum
        temphist[bin] += W[k] * Mag[k]


    # 高斯平滑
    temp = [temphist[BinNum - 1], temphist[BinNum - 2], temphist[0], temphist[1]]
    temphist.insert(0, temp[0])
    temphist.insert(0, temp[1])
    temphist.insert(len(temphist), temp[2])
    temphist.insert(len(temphist), temp[3])      # padding

    hist = []
    for i in range(BinNum):
        hist.append((temphist[i] + temphist[i+4]) * (1.0 / 16.0) + (temphist[i+1] + temphist[i+3]) * (4.0 / 16.0) + temphist[i+2] * (6.0 / 16.0))
    maxval = max(hist)

    return maxval,hist

def downsample(img,step = 2):
    return img[::step,::step]

def LocateKeyPoint(DoG,sigma,GuassianPyramid,n,BinNum = 36,contrastThreshold = 0.04,edgeThreshold = 10.0):
    SIFT_ORI_SIG_FCTR = 1.52
    SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR
    SIFT_ORI_PEAK_RATIO = 0.8

    SIFT_FIXPT_SCALE = 1

    KeyPoints = []
    O = len(DoG)
    S = len(DoG[0])
    for o in range(O):
        for s in range(1,S-1):
            threshold = 0.5*contrastThreshold/(n*255*SIFT_FIXPT_SCALE)
            img_prev = DoG[o][s-1]
            img = DoG[o][s]
            img_next = DoG[o][s+1]
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    val = img[i,j]
                    eight_neiborhood_prev = img_prev[max(0, i - 1):min(i + 2, img_prev.shape[0]), max(0, j - 1):min(j + 2, img_prev.shape[1])]
                    eight_neiborhood = img[max(0, i - 1):min(i + 2, img.shape[0]), max(0, j - 1):min(j + 2, img.shape[1])]
                    eight_neiborhood_next = img_next[max(0, i - 1):min(i + 2, img_next.shape[0]), max(0, j - 1):min(j + 2, img_next.shape[1])]
                    if np.abs(val) > threshold and \
                        ((val > 0 and (val >= eight_neiborhood_prev).all() and (val >= eight_neiborhood).all() and (val >= eight_neiborhood_next).all())
                         or (val < 0 and (val <= eight_neiborhood_prev).all() and (val <= eight_neiborhood).all() and (val <= eight_neiborhood_next).all())):

                        point,x,y,layer = adjustLocalExtrema(DoG,o,s,i,j,contrastThreshold,edgeThreshold,sigma,n,SIFT_FIXPT_SCALE)
                        if point == None:
                            continue

                        scl_octv = point[-1]*0.5/(1 << o)
                        omax,hist = GetMainDirection(GuassianPyramid[o][layer],x,y,int(np.round(SIFT_ORI_RADIUS * scl_octv)),SIFT_ORI_SIG_FCTR * scl_octv,BinNum)
                        mag_thr = omax * SIFT_ORI_PEAK_RATIO
                        for k in range(BinNum):
                            if k > 0:
                                l = k - 1
                            else:
                                l = BinNum - 1
                            if k < BinNum - 1:
                                r2 = k + 1
                            else:
                                r2 = 0
                            if hist[k] > hist[l] and hist[k] > hist[r2] and hist[k] >= mag_thr:
                                bin = k + 0.5 * (hist[l]-hist[r2]) /(hist[l] - 2 * hist[k] + hist[r2])
                                if bin < 0:
                                    bin = BinNum + bin
                                else:
                                    if bin >= BinNum:
                                        bin = bin - BinNum
                                temp = point[:]
                                temp.append((360.0/BinNum) * bin)
                                KeyPoints.append(temp)
    return KeyPoints

def SIFT(img,showDoGimgs = False):
    SIFT_SIGMA = 1.6
    SIFT_INIT_SIGMA = 0.5
    sigma0 = np.sqrt(SIFT_SIGMA**2-SIFT_INIT_SIGMA**2)
    n = 3
    DoG,GuassianPyramid = getDoG(img, n,sigma0)
    KeyPoints = LocateKeyPoint(DoG, SIFT_SIGMA, GuassianPyramid, n)
    return KeyPoints

def run(img1):
    #img1 = cv2.imread("1.jpg")
    if len(img1.shape) == 3:
        img = img1.mean(axis=-1)
    else:
        img = img1
    keyPoints = SIFT(img)
    for k in keyPoints:
        cv2.circle(img1, (int(k[0]), int(k[1])), 2, (0,0,255), 1)
    return img1
