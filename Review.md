<img width="330" height="31" alt="image" src="https://github.com/user-attachments/assets/3a8af7ae-ddb6-4115-888e-2e851f25471c" />#<h1 align="center">强化学习笔记</h1>
#<h1 align="left">M1基础</h1>
#1.强化学习与有监督学习的区别，两者目标不同
#<img width="1172" height="281" alt="image" src="https://github.com/user-attachments/assets/3b5fad11-bfeb-43ca-8504-5822c72da814" />
<img width="821" height="250" alt="image" src="https://github.com/user-attachments/assets/618a2363-5e1c-4af2-9647-54e785dc423d" />
<img width="830" height="327" alt="image" src="https://github.com/user-attachments/assets/55626caf-0830-4c2c-9273-36c71f7b5941" />

#2.强化学习中的占用度量
</br>
<img width="568" height="307" alt="image" src="https://github.com/user-attachments/assets/648b438e-c655-4e6f-b81d-eb873b3ab9d4" />
</br>
#3.强化学习中环境的下一刻状态的概率分布将由当前状态和智能体的动作来共同决定，用最简单的数学公式表示则是
</br>
<img width="391" height="50" alt="image" src="https://github.com/user-attachments/assets/b90e8286-8386-4c5f-a05d-669c140924b8" />
#<h1 align="left">M2多臂老虎机</h1>
多臂老虎机可以看作一种简化版的强化学习，他不存在状态信息，只有动作和奖励，学习这个有利于理解强化学习
#<h2 align="left">M2.1问题介绍</h2>
1.问题定义，K根拉杆的老虎机，每一个杆子都对应一个未知的奖励分布，从头尝试，目的是T次拉杆之后获得最多的奖励。奖励分布是未知的，所以需要衡量“探索概率”还是“根据经验选择获奖次数多的”
<img width="587" height="607" alt="image" src="https://github.com/user-attachments/assets/cf0ffc90-fe23-4dea-b34b-460476092d5a" />
</br>
2.形式化描述
</br>
<img width="592" height="322" alt="image" src="https://github.com/user-attachments/assets/9aae82a5-38ef-43d3-aa9a-5dfc7e55610d" />
<img width="382" height="47" alt="image" src="https://github.com/user-attachments/assets/e7e540c7-d42c-4205-b369-5b86bf5f8464" />
</br>
<img width="671" height="267" alt="image" src="https://github.com/user-attachments/assets/1aad1398-695e-46c9-b824-26b4750f0596" />
<img width="808" height="556" alt="image" src="https://github.com/user-attachments/assets/0c8f6928-e51b-4bdc-a5e4-0997e478bbee" />
</br>
</br>
这个概率分布R，具体内容为执行动作a,不同奖励产生的概率
</br>
3.累积懊悔
</br>
<img width="563" height="316" alt="image" src="https://github.com/user-attachments/assets/004396be-41a4-4d78-b50b-037c0d7b5426" />
</br>
########
4.重点： 估计期望奖励
</br>
<img width="593" height="175" alt="image" src="https://github.com/user-attachments/assets/5b8b5f9f-8d58-4c71-ad4d-1245d4ccaf8a" />
</br>
这里可以理解为<img width="47" height="37" alt="image" src="https://github.com/user-attachments/assets/85fdf20e-061c-498b-a4f7-3b7566ffa371" />为平均奖励，k轮后的平均奖励，这里可理解为前k-1轮的奖励再加上k轮单独的奖励除以k轮，将每一轮的奖励平均到每一轮
</br>
编写代码来实现一个拉杆数为 10 的多臂老虎机。其中拉动每根拉杆的奖励服从伯努利分布（Bernoulli distribution），即每次拉下拉杆有p的概率获得的奖励为 1，有的1-p概率获得的奖励为 0。奖励为 1 代表获奖，奖励为 0 代表没有获奖。
</br>
<img width="973" height="777" alt="image" src="https://github.com/user-attachments/assets/3904f9fc-98fc-40d4-bcea-cd5697eae45b" />
</br>
代码解释1：self.best_idx = np.argmax(self.probs) argmax返回的是编号
</br>
代码解释2：self.best_prob = self.probs[self.best_idx]  通过索引来获得最大概率
</br>
代码解释3：if np.random.rand() < self.probs[k]: return 1  功能：生成一个 [0,1) 的随机浮点数用来模拟一次随机试验
概率大于随机数即中奖
</br>
5.重点：E-贪心策略
</br>
探索(exploration):是指尝试拉动更多可能的拉杆，这根拉杆不一定会获得最大的奖励，但这种方案能够摸清楚所有拉杆的获奖情况。例如，对于一个 10 臂老虎机，我们要把所有的拉杆都拉动一下才知道哪根拉杆可能获得最大的奖励。
</br>
利用(exploitation)：是指拉动已知期望奖励最大的那根拉杆，由于已知的信息仅仅来自有限次的交互观测，所以当前的最优拉杆不一定是全局最优的。例如，对于一个 10 臂老虎机，我们只拉动过其中 3 根拉杆，接下来就一直拉动这 3 根拉杆中期望奖励最大的那根拉杆，但很有可能期望奖励最大的拉杆在剩下的 7 根当中，即使我们对 10 根拉杆各自都尝试了 20 次，发现 5 号拉杆的经验期望奖励是最高的，但仍然存在着微小的概率—另一根 6 号拉杆的真实期望奖励是比 5 号拉杆更高的。
</br>
完全贪婪算法即在每一时刻采取期望奖励估值最大的动作（拉动拉杆），这就是纯粹的利用，而没有探索，所以我们通常需要对完全贪婪算法进行一些修改，其中比较经典的一种方法为<img width="11" height="18" alt="image" src="https://github.com/user-attachments/assets/15418530-00a9-4f65-8d44-8d204336a53c" />-贪婪算法。该算法在完全贪婪算法的基础上增加了噪声，每次以概率1-E选择以往经验中期望奖励估值最大的那个拉杆（利用），以概率E随机选择一根拉杆进行探索，公式如下：
</br>
<img width="410" height="87" alt="image" src="https://github.com/user-attachments/assets/1c9ddfb5-3103-452b-a219-e591947ca17f" />
</br>
<img width="483" height="217" alt="image" src="https://github.com/user-attachments/assets/ef65ba92-067b-405b-9835-ba3ef804052b" />
</br>
#<h1 align="left">M3马尔可夫决策过程</h1>
#<h2 align="left">M3.1马尔可夫过程</h2>
1.随机过程
</br>
<img width="492" height="281" alt="image" src="https://github.com/user-attachments/assets/5b8d70d0-11ff-4317-9af6-a9ab5176d0a1" />
</br>
t+1时刻某状态及其其发生概率取决于1~t时刻的状态
</br>
2.马尔可夫性质
</br>
<img width="503" height="368" alt="image" src="https://github.com/user-attachments/assets/40513965-ea5e-4282-abe6-572dcaadc6f6" />
</br>
t+1时刻的状态只与t时刻有关但是这不是指与历史无关因为t中包含了t-1.t-1包含了t-2
</br>
3.马尔可夫过程
</br>
<img width="952" height="507" alt="image" src="https://github.com/user-attachments/assets/1b8cf43f-0067-490d-aab5-6d217f47911f" />
</br>
<img width="822" height="710" alt="image" src="https://github.com/user-attachments/assets/992b32ab-a49a-46f5-b3db-0a189b0583b5" />
</br>
根据状态转移矩阵写出的序列，这个过程也叫做采样，第n行就代表状态n。重点状态无出发
</br>
#<h2 align="left">M3.2马尔可夫奖励过程-MRP</h2>
</br>
<img width="476" height="457" alt="image" src="https://github.com/user-attachments/assets/bc5f2de9-88b8-42da-a5fe-a9568b422612" />
</br>
接近1的折扣因子关注长期累计奖励，接近0的折扣因子考虑短期奖励
1.某点的回报：从某点开始出发每个状态点的奖励乘以折扣之和
</br>
<img width="493" height="176" alt="image" src="https://github.com/user-attachments/assets/31f5feb4-320c-47dd-bea1-f20b63f40430" />
</br>
<img width="508" height="541" alt="image" src="https://github.com/user-attachments/assets/4a99959a-1405-4295-a3ab-308001e0c419" />
</br>
2.价值函数
价值（value）：在马尔可夫奖励过程中，一个状态的期望回报（即从这个状态出发的未来累积奖励的期望）被称为这个状态的价值（value）。所有状态的价值就组成了价值函数（value function），价值函数的输入为某个状态，输出为这个状态的价值。
价值就是累积的回报的期望值公式为：
</br>
<img width="503" height="396" alt="image" src="https://github.com/user-attachments/assets/d210ac64-0974-40d5-8a4d-11d80a6d2f66" />
</br>
公式理解：某点的价值等于[本状态的奖励+折扣因子X上一个状态的价值]即
贝尔曼方程[***无敌重点***]
</br>
<img width="1206" height="1514" alt="eee4d4a883563cadeb96247716b6bd5a" src="https://github.com/user-attachments/assets/49c6dc4f-5f00-4c9f-a160-459aa0f4f96e" />
</br>
<img width="1206" height="1005" alt="eb710c4a844057a556f30cc947cb404c" src="https://github.com/user-attachments/assets/88a0a379-61dc-4c5e-adf9-4f8b484fa9d8" />
</br>
上面两图为贝尔曼方程实操，方便理解*Q动作方程不用乘以概率
<img width="487" height="380" alt="image" src="https://github.com/user-attachments/assets/901630a9-bce6-4a0d-a926-ac87ae3aab7a" />
</br>
贝尔曼方程：本状态价值=本状态的奖励+折扣因子X从本状态出发到达所有点的概率X本状态后继的价值
#<h1 align="left">M3.3马尔可夫决策过程</h1>
#<h2 align="left">定义</h2>
如果在MRP过程中加入刺激（agent的动作）那么叫做MDP
</br>
<img width="330" height="31" alt="image" src="https://github.com/user-attachments/assets/8fc01e5f-c8ba-4aea-85ed-63d3d500c9da" />构成
</br>
<img width="570" height="228" alt="image" src="https://github.com/user-attachments/assets/43f06ff1-3de3-454f-80da-d1701b8d53d6" />
</br>
不再使用类似 MRP 定义中的状态转移矩阵方式，而是直接表示成了状态转移函数。（即P函数，表示在状态s下执行a动作到达s2概率）

