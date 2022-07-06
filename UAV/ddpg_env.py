import math
import random
import numpy as np

# B-UAV 坐标
vmax = 150
slot = 1
N = 50 #  ?
lmax = vmax*slot
low0 = 1.42E-04
delta = -174
D = 0.2
rin = 0.00625  #传输速率
alpha = 0.5
#任务数量
B = 1
P1 = P2 = -11

B0 = [1000, 1000, 500]
B1 = [3000, 1000, 500]
B2 = [5000, 1000, 500]
B3 = [1000, 3000, 500]
B4 = [3000, 3000, 500]
B5 = [5000, 3000, 500]
B6 = [1000, 5000, 500]
B7 = [3000, 5000, 500]
B8 = [5000, 5000, 500]

M = [B0,B1,B2,B3,B4,B5,B6,B7,B8]

tcom = [0,0,0,0,0,0,0,0,0]; tin = [0,0,0,0,0,0,0,0,0]; ri = [0,0,0,0,0,0,0,0,0]; snri = [0,0,0,0,0,0,0,0,0]; hi = [0,0,0,0,0,0,0,0,0]; di = [0,0,0,0,0,0,0,0,0]

task0 = [];task1 = [];task2 = [];task3 = [];task4 = [];task5 = [];task6 = [];task7 = [];task8 = []
T_traj = []
sn = []
for _ in range(5):
    task0 = task0 + [0.5, 0.6, 0.8, 1.6, 0.7, 1.3, 1.2, 1.1, 0.7, 0.8]
    task1 = task1 + [0.8, 1.6, 1.2, 1.5, 0.9, 1.3, 1.2, 1.4, 0.6, 0.9]
    task2 = task2 + [0.7, 0.6, 1.4, 1.2, 1.5, 0.8, 0.9, 1.3, 1.1, 0.8]
    task3 = task3 + [1.2, 1.3, 1.4, 1.1, 1.6, 1.5, 0.6, 1.5, 0.7, 0.5]
    task4 = task4 + [1.3, 0.7, 1.6, 1.2, 1.5, 1.5, 1.2, 1.1, 0.9, 0.5]
    task5 = task5 + [1.1, 0.6, 0.7, 0.7, 1.6, 1.2, 1.5, 1.5, 1.2, 0.6]
    task6 = task6 + [0.9, 0.6, 0.7, 1.5, 1.2, 1.1, 0.9, 0.5, 1.3, 0.5]
    task7 = task7 + [1.0, 0.7, 0.7, 0.6, 1.5, 1.2, 1.1, 0.9, 1.2, 1.3]
    task8 = task8 + [1.7, 0.5, 0.3, 0.4, 1.2, 0.6, 0.6, 1.5, 0.5, 1.3]
    #T_traj = T_traj + [119,  26,  96,  89,  86,  87,  51,  62.,  48,  127]
    #sn = sn + [9, 8, 5, 6, 4, 3, 11, 3, 6, 7]

sita = [0*math.pi, 0.5*math.pi, 1*math.pi, 1.5*math.pi]

#  action = sita, dist, sn
dist = [0., 150.]

sn = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#action = [[x,y,z]for x in sita for y in dist for z in sn]

#print(action)

bin = lambda  x,y:x>=y #bin
bsub = lambda  x,y:x-y #need to T

class enva(object):
    def __init__(self):
        self.ri = 0
        self.op2 = 0
        self.xn = 3000
        self.yn = 3000
        self.h = 1500

    def reset(self):
        self.ri = 0
        self.op2 = 0
        self.xn = 3000
        self.yn = 3000
        self.h = 1500
        b1sum = task0[0];b2sum = task1[0] ;b3sum = task2[0] ;b4sum = task3[0] ;b5sum = task4[0]
        b6sum = task5[0] ;b7sum = task6[0] ; b8sum = task7[0] ; b9sum = task8[0]
        return np.array([b1sum, b2sum, b3sum,
                         b4sum, b5sum, b6sum,
                         b7sum, b8sum, b9sum, self.xn, self.yn])

    def step(self, s, a, h):
        # 每一幕要经历五十个时隙
        done = True
        b1sum, b2sum, b3sum, b4sum, b5sum, b6sum, b7sum, b8sum, b9sum, xn, yn = s
        bsum = [b1sum, b2sum, b3sum, b4sum, b5sum, b6sum, b7sum, b8sum, b9sum]
        bflag = [bsub(b1sum,1),bsub(b2sum,1),bsub(b3sum,1),bsub(b4sum,1),bsub(b5sum,1),bsub(b6sum,1),bsub(b7sum,1),bsub(b8sum,1),bsub(b9sum,1)]
        Bsub = [0,0,0,0,0,0,0,0,0]
        T_comMax = 0
        #动作:方向距离，虚拟机数量
        for each in range(9):
            di[each] = math.sqrt((xn - M[each][0]) ** 2 + (yn - M[each][1]) ** 2 + (1500 - M[each][2]) ** 2)
            snri[each] = (low0 * 20) / (-174) * B / 9 * di[each] ** 2
            ri[each] = (B / 9) * np.log2(1 + abs(snri[each]))
            if(bflag[each]):
                Bsub[each] = bsub(bsum[each], 1)
            tin[each] = (Bsub[each] * bflag[each] / ri[each])

            tcom[each] = (Bsub[each] * bflag[each] / rin)

            T_comMax = max(tcom[each], T_comMax)

        T_tran = np.sum(tin)

        snchoice = min(a[2], (T_comMax+T_tran)/rin)
        #根据任务量选择飞行方向和虚拟机数量
        xn_ = xn + max(math.cos(a[0])*a[1], xn)
        yn_ = yn + max(math.sin(a[0])*a[1], yn)

        print('h:', h, ' xn:', xn_, 'yn', yn_, 'sn', snchoice)

        self.op2 = -((T_tran + alpha * T_comMax) / 9)

        if (xn_ < 0 or xn_ > 6000 or yn_ < 0 or yn_ > 6000):
            self.op2 -= P1
        if (snchoice > 10):
            self.op2 -= P2
        s_ =  np.array([task0[h], task1[h], task2[h],
                         task3[h], task4[h], task5[h],
                         task6[h], task7[h], task8[h], xn_, yn_])

        return s_, self.op2, done