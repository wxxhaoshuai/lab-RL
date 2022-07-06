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
rin = 62.5  #传输速率
alpha = 0.5
#任务数量
B = 1

P1 = -11
P2 = -11
#tasknum = np.random.randint(5,7,10)
sita = np.array([0*math.pi,0.5*math.pi,1*math.pi,1.5*math.pi])

print(sita)
B0 = [1000, 1000, 500]
B1 = [3000, 1000, 500]
B2 = [5000, 1000, 500]
B3 = [1000, 3000, 500]
B4 = [3000, 3000, 500]
B5 = [5000, 3000, 500]
B6 = [1000, 5000, 500]
B7 = [3000, 5000, 500]
B8 = [5000, 5000, 500]
# T-UAV T-UAV以选择的轨迹飞行，以最小化系统的传输延迟。
M = [B0,B1,B2,B3,B4,B5,B6,B7,B8]

T = [3000., 3000., 1500.]

Xn = Yn = np.zeros((50,4),dtype=float)
Hn = 1500
temp1 = T[0]
temp2 = T[1]
'''for i in range(3):
    for j in range(49):
        T[0] += T[0] * math.cos((sita[i]))
        if(T[0]==0):print(T[0])
        #Xn[i][j].append(T[0])
        T[1] += T[1] * math.sin((sita[i]))
        if (T[1] == 0): print(T[0])
        #Yn[i][j].append(T[1])
'''
#距离这一块有点问题


#Qnt = np.vstack((np.vstack((Xn,Yn)),Hn))
tcom = tin = ri = snri = hi = di = np.zeros(9)

pin = np.full(20,9)

tasklist = np.array([6.09762701, 6.43037873, 6.20552675, 6.08976637, 5.8473096,  6.29178823,
 5.87517442, 6.783546,   6.92732552, 5.76688304, 6.58345008, 6.05778984,
 6.13608912, 6.85119328, 5.14207212, 5.1742586,  5.04043679, 6.66523969,
 6.5563135,  6.7400243,  6.95723668, 6.59831713, 5.92295872, 6.56105835,
 5.23654885, 6.27984204, 5.28670657, 6.88933783, 6.04369664, 5.82932388,
 5.52911122, 6.54846738, 5.91230066, 6.1368679,  5.0375796,  6.23527099,
 6.22419145, 6.23386799, 6.88749616, 6.3636406,  5.7190158,  5.87406391,
 6.39526239, 5.12045094, 6.33353343, 6.34127574, 5.42076512, 5.2578526,
 5.6308567,  5.72742154, 6.14039354, 5.87720303,6.97674768, 5.20408962,
 5.41775351, 5.32261904, 6.30621665, 5.50658321, 5.93262155, 5.48885118,
 5.31793917, 5.22075028, 6.31265918, 5.2763659,  5.39316472, 5.73745034,
 6.64198646, 5.19420255, 6.67588981, 5.19219682, 6.95291893, 5.9373024,
 6.95352218, 6.20969104, 6.47852716, 5.07837558, 5.56561393, 5.24039312,
 5.5922804, 5.23745544, 5.63596636, 5.82852599, 5.12829499, 6.38494424,
 6.13320291, 5.53077898, 6.04649611, 5.18788102, 6.15189299, 6.8585924,
 5.6371379,  6.33482076, 5.26359572, 6.43265441, 5.57881219, 5.36638272,
 6.17302587, 5.04021509, 6.65788006, 5.00939095, 6.35563307, 5.54001595,
 6.47038804, 6.92437709, 5.49750629, 6.15231467, 6.18408386, 6.14450381,
 5.44616327, 6.90549802,5.89425076, 6.69281734, 6.39895855, 5.5948739,
 6.62759564, 5.79301148, 6.76220639, 6.16254575, 6.76347072, 6.38506318,
 6.45050856, 6.00264876, 6.91216727, 6.2879804,  5.8477101,  6.21278643,
 5.0383864,  5.60314963, 6.32034707, 5.58015521, 6.23603086, 5.8575374,
 5.27094813, 5.59656465, 6.13992982, 6.18174552, 6.1486505,  6.30640164,
 6.30420654, 5.86283687, 6.79309319, 5.73512374, 5.87172985, 6.78384671,
 6.61238798, 6.40777717, 5.20045377, 6.83896523, 6.4284826,  6.99769401,
 5.29889661, 6.73625211, 5.32498587, 6.23111913, 5.24763997, 6.69601646,
 6.61463792, 6.13820148, 5.81436659, 5.13833399, 6.39485755, 5.90708537,
 6.4441112,  6.73276465, 6.95104301, 6.71160668, 5.02342817, 5.71995613,
 6.45998112, 5.34325935, 6.04207321, 5.10867598, 5.39999305, 5.03704359,
 6.58739541, 5.44784938, 5.69070336, 6.85616259, 6.4088288,  5.06367786,
 5.32938831, 6.2429568,  6.15445718, 5.47578564, 6.868428,   6.22793191,
 6.07126561, 6.17981995, 6.46024406, 5.62388999, 5.79644212, 5.4196875,
 5.37238601, 6.88874478, 6.47910159, 5.98091762, 5.45482926, 5.50871296,
 5.11605832, 5.86883325])

bin = lambda  x,y:x>=y
bsub = lambda  x,y:x-y

### 接入用户请求的环境，这样设置不知道合不合理
b0task = np.random.choice(tasklist, 170)
b0flag = bin(int(np.sum(b0task)), 1024)
b1task = np.random.choice(tasklist, 170)
b1flag = bin(int(np.sum(b1task)), 1024)
b2task = np.random.choice(tasklist, 170)
b2flag = bin(int(np.sum(b2task)), 1024)
b3task = np.random.choice(tasklist, 170)
b3flag = bin(int(np.sum(b3task)), 1024)
b4task = np.random.choice(tasklist, 170)
b4flag = bin(int(np.sum(b4task)), 1024)
b5task = np.random.choice(tasklist, 170)
b5flag = bin(int(np.sum(b5task)), 1024)
b6task = np.random.choice(tasklist, 170)
b6flag = bin(int(np.sum(b6task)), 1024)
b7task = np.random.choice(tasklist, 170)
b7flag = bin(int(np.sum(b7task)), 1024)
b8task = np.random.choice(tasklist, 170)
b8flag = bin(int(np.sum(b8task)), 1024)

bsub0 = bsub(int(np.sum(b0task)), 1024)
bsub1 = bsub(int(np.sum(b1task)), 1024)
bsub2 = bsub(int(np.sum(b2task)), 1024)
bsub3 = bsub(int(np.sum(b3task)), 1024)
bsub4 = bsub(int(np.sum(b4task)), 1024)
bsub5 = bsub(int(np.sum(b5task)), 1024)
bsub6 = bsub(int(np.sum(b6task)), 1024)
bsub7 = bsub(int(np.sum(b7task)), 1024)
bsub8 = bsub(int(np.sum(b8task)), 1024)
bflag = [b0flag, b1flag, b2flag, b3flag, b4flag, b5flag, b6flag, b7flag, b8flag]
bsub = [bsub0, bsub1, bsub2, bsub3, bsub4, bsub5, bsub6, bsub7, bsub8]

snMAx = np.sum(bflag)
#print(snMAx)
sn = np.random.randint(0, int(snMAx) + 1, 50)
dista = [119.34364446,  26.15004935,  96.87412674,  89.57864353,  86.51836235,
  87.74378817,  51.60240446,  62.35792915,  48.3425442,  127.0094393 ]

Qnt = np.vstack((Xn,Yn))
#所有动作空间
action = [[degree,dist,Sn]for degree in sita for dist in dista for Sn in range(9)]
print(len(action))
class enva(object):
    def __init__(self):
        self.reward = 0
        self.op1 = 0
        self.op2 = 0
    #刷新
    def reset(self):
        #刷新系统的状态
        xn = 3000
        yn = 3000
        self.op1 = 0
        self.op2 = 0
        return np.array([0,0,0,
                      0,0,0,
                      0,0,0,xn,yn])

    #每一个时隙每一步,t时隙
    def step(self, s, a, h):
        print('a:',a)
        done = True
        xy = s[-2:]
        Qn = [random.choice(Qnt[0]),random.choice(Qnt[1]),random.choice(Qnt[2])]
        T_comMAx = 0
        for each in range(9):

            di[each] = math.sqrt((Qn[0] - M[each][0]) ** 2 + (Qn[1] - M[each][1]) ** 2 + (Qn[2] - M[each][2]) ** 2)
            # hi[each] = low0/(Qn[0] - M[each][0]) ** 2 + (Qn[1] - M[each][1]) ** 2 + (Qn[2] - M[each][2]) ** 2
            snri[each] = (low0 * pin[each]) / (-174) * B / 9 * di[each] ** 2
            ri[each] = (B / 9) * np.log2(1 + abs(snri[each]))
            tin[each] = (bsub[each] * bflag[each] / ri[each])

            # T-com
            tcom[each] = (bsub[each] * bflag[each] / rin)

            T_comMAx = max(tcom[each], T_comMAx)

        T_tran = np.sum(tin)
        #有点问题
        snchoice = action[a][2]
        print(snchoice)

        xn = xy[0]

        yn = xy[1]

        xn_ = max(0, (xn + max(lmax * math.cos(action[a][0]), xn)))
        yn_ = max(0, (yn + max(lmax * math.sin(action[a][0]), yn)))
        # out of range

        print('h:', h, ' xn:', xn_, 'yn', yn_)


        self.op1 = T_comMAx * ((1 + D) ** (snchoice - 1))
        self.op2 = -((T_tran+alpha*T_comMAx)/snMAx)

        if(xn_<0 or xn_>6000 or yn_<0 or yn_>6000):
            self.op2 -= P1
        if(snchoice>snMAx):
            self.op2 -= P2

        s_ = np.array([np.sum(b0task),np.sum(b1task),np.sum(b2task),
                      np.sum(b3task),np.sum(b4task),np.sum(b5task),
                      np.sum(b6task),np.sum(b7task),np.sum(b8task),xn_,yn_])

        return s_ ,self.op2,done