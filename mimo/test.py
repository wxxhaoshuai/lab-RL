from dqn.RL_brain import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt

def reset():
    Bmax=0.2500#最大电量
    H_=np.random.randn(2,2)#2*2正态分布的随机信道
    a=np.maximum(H_,0)#最大的信道
    b=np.where(a>0,1,0)#二值化
    c,d=np.linalg.eig(H_*H_.T)#求信道矩阵(2*2)的特征值与特征向量非零特征值可以用来代表各并行独立信道在时隙t 时的信道状态
    h=str(b[0][0])+str(b[0][1])+str(b[1][0])+str(b[1][1])#对应下文的“xxxx”
    #状态三元组，信道，电量，能量
    state1=np.array([h,min(np.random.choice(Blist),Bmax),np.random.choice(Elist)])
    return state1,c

def update():
    bili1=1;bili2=2#比例？

    Elist=[0,0.05,0.1,0.15,0.2]
    Blist=[0,0.05,0.10,0.15,0.20,0.25]
    Emax=0.2;Bmax=0.2500
    number=50000#总数
    observation=reset()[0]#状态
    print(np.shape(observation))
    battery=observation[1]
    print(battery)
    while number>0:
        number-=1
        print(number)
        #动作,找到每次电池的电量
        #在这个系统中，电池将收集到的能量存储在电池中，接着发射端利用电池中的能量
        #向发射端传送功率
        #Pt ，所以在电池中存在能量转移的过程。电池中的能量应该遵循如下
        #根据电量选择策略
        actions=list(range(len(Blist[0:Blist.index(float(battery))+1])))#返回的是
        print("actions", actions)
        action=QL.choose_action(observation)/20#为什么要除20？
        print("action", action)
        if  action*20 not in actions:
            #新动作
            reward=-101
            # Bt+1
            battery=max(min(round(float(observation[2]),2),Bmax),0)
            print("battery",battery)
        else:
            p1_=action*(bili1/(bili1+bili2))
            p2_=action*(bili2/(bili1+bili2))
            print("p1,p2", (p1_,p2_))
            _,e=reset()

            reward=np.log2(1+(e[0]*p1_*float(battery))/2)+np.log2(
                1+(e[1]*p2_*float(battery))/2)#当前选取行动所获得的奖励
            print("reward",reward)
            battery=max(min(round(float(battery)-action+np.random.choice(Elist),2),Bmax),0)
            print("battery", battery)
        for i in range(6):#电池能量
            if abs(Blist[i] - battery) <= 0.02:
                battery = Blist[i]

                observation_ = np.array([reset()[0][0], battery, reset()[0][2]])
                QL.store_transition(observation, action * 20, reward, observation_)
                observation = observation_


Htuple=['0000','1000','0100','0010','0001','1100','1010','1001','0110','0011',
'0101','1110','1101','0111','1011','1111']
Elist=[0,0.05,0.1,0.15,0.2]
Blist=[0,0.05,0.10,0.15,0.20,0.25]
Bmax=0.2500
bili1=1;bili2=2
rq=0
y1=[]
#6个电量对应的动作，三个输入
QL=DeepQNetwork(6,3,learning_rate=0.01,reward_decay=0.9,
e_greedy=0.9,replace_target_iter=200,memory_size=2000,output_graph=False)
update()
state=np.array(["0000",0.25,0.05])
#测试
for num in range(200):
    action=QL.choose_action(state)/20
    p1_=action*(bili1/(bili1+bili2))
    p2_=action*(bili2/(bili1+bili2))
    battery=float(state[1])
    print(battery)
    battery=max(min(round(battery-action+np.random.choice(Elist),2),Bmax),0)
    print("battery", battery)
    print(battery)
    H_=np.random.randn(2,2)
    a=np.maximum(H_,0)
    b=np.where(a>0,1,0)
    h=str(b[0][0])+str(b[0][1])+str(b[1][0])+str(b[1][1])
    print("h", h)
    c,d=np.linalg.eig(H_*H_.T)#c为两个特征值
    reward=np.log2(1+(c[0]*p1_*battery)/2)+np.log2(1+(c[1]*p2_*battery)/2)

    rq=rq+reward*(0.9**num)
    for i in range(6):
        if abs(Blist[i]-battery)<=0.02:
            battery=Blist[i]
            print("battery", battery)
            state=np.array([h,battery,np.random.choice(Elist)])
            print("state", state)

    y1.append(rq)

    print(y1)
x=np.arange(200)
plt.title('DQN')
plt.xlabel('time')
plt.ylabel('totalreward')
plt.plot(x,y1)
plt.show()
