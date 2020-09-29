import numpy as np
import pandas as pd
import talib

def get_l2_dist(u1, u2):
    p1, p2 = np.array(u1), np.array(u2)
    return np.linalg.norm(p2-p1, ord=2)


def get_stability(image, bdt, width=3, r_max=2, neighbours_min=3):
    tmin = np.min(image) + 2
    tmax = np.max(image) - 2
    npix = image.size
    dist_lst = []
    for t in range(tmin, tmax):
        # current edge
        idx = np.where(bdt[t - tmin].reshape(image.shape) == 1)
        tmp1, tmp2 = idx
        idx = list(zip(tmp1, tmp2))
        windows = np.zeros((width,npix))

        for w in range(1,width+1):
            dist = np.zeros(image.shape)
            # for each point in the current edge, find the neighbours in the next edge.
            for (x, y) in idx:
                if (x == 0) or (y == 0) or (x == image.shape[0] - 1) or (y == image.shape[1] - 1):
                    continue
                r = 1
                neighbour = []
                while len(neighbour) < neighbours_min and r < r_max:
                    try:
                        for m in range(x - r, x + r + 1):
                            for n in range(y - r, y + r + 1):
                                if (m<=0) or (n<=0) or (m>=image.shape[0]-1) or (n>=image.shape[1]-1) or (t-tmin+w>=len(bdt)):
                                    continue
                                if bdt[t - tmin + w][m, n] == 1:
                                    neighbour.append((m, n))
                    except Exception as e:
                        print(e)
                        print(m, n)
                    r += 1
                if len(neighbour) < neighbours_min:
                    # something large enough
                    dist[x, y] = r_max
                    continue
                # Calculate the mean value of the position
                xo, yo = zip(*neighbour)
                xo, yo = np.mean(xo), np.mean(yo)
                dist[x, y] = get_l2_dist((x, y), (xo, yo))
            windows[w-1,] = dist.flatten()
        dist_lst.append(np.diff(windows.T,2))
        
    output = list(map(lambda x: np.sum(np.abs(x),axis=1).reshape(image.shape), dist_lst))
    return output


def min_max(lst):
    epsilon = 1e-9
    return (lst-np.min(lst)) / (np.max(lst)-np.min(lst) + epsilon)

    
def pix_scale(stab,bdts):
    output = []
    epsilon = 1e-9
    for i in range(len(stab)):
        tmp = np.sum(stab[i]) / (np.where(bdts[i]==1)[0].size+epsilon)
        output.append(tmp)
    return output


def get_emastab(stab,period=14):
    testema = talib.EMA(np.array(stab),period)
    return pd.DataFrame(testema).fillna(method='bfill').values