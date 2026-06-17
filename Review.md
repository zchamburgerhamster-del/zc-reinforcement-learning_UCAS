<img width="122" height="33" alt="image" src="https://github.com/user-attachments/assets/81d5cdf8-e87b-477b-be33-bf9bf778eb4d" /><img width="330" height="31" alt="image" src="https://github.com/user-attachments/assets/3a8af7ae-ddb6-4115-888e-2e851f25471c" />#<h1 align="center">强化学习笔记</h1>
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
#<h2 align="left">M3.3马尔可夫决策过程</h2>
#<h3 align="left">定义</h3>
如果在MRP过程中加入刺激（agent的动作）那么叫做MDP
</br>
<img width="330" height="31" alt="image" src="https://github.com/user-attachments/assets/8fc01e5f-c8ba-4aea-85ed-63d3d500c9da" />构成
</br>
<img width="570" height="228" alt="image" src="https://github.com/user-attachments/assets/43f06ff1-3de3-454f-80da-d1701b8d53d6" />
</br>
不再使用类似 MRP 定义中的状态转移矩阵方式，而是直接表示成了状态转移函数。（即P函数，表示在状态s下执行a动作到达s2概率）
 </br>
 <img width="360" height="223" alt="image" src="https://github.com/user-attachments/assets/760722f0-1c14-42a0-bfad-9492837aeac6" />
</br>
如上图所示为一个agent和环境MDP交互过程，agent根据当前状态St选择一个动作At，然后环境反馈一个下一状态St+1（通过状态转移函数），返回一个本轮的奖励Rt
#<h3 align="left">3.3.1策略（策略函数）</h3>
智能体的策略（Policy）通常用字母<img width="12" height="22" alt="image" src="https://github.com/user-attachments/assets/d34b1945-8584-4fad-8f04-8e2602fd94d4" />
1.表示在输入状态s情况下采取动作a的概率。当一个策略是确定性策略（deterministic policy）时，它在每个状态时只输出一个确定性的动作，即只有该动作的概率为 1，其他动作的概率为 0；当一个策略是随机性策略（stochastic policy）时，它在每个状态时输出的是关于动作的概率分布，然后根据该分布进行采样就可以得到一个动作。
2.MDP中策略只与当前状态有关，不需要考虑历史状态
3.MRP中价值函数与概率和上一状态有关
4.但是MDP中价值函数和动作有关，分为状态价值函数与动作价值函数
#<h3 align="left">3.3.2状态价值函数-与MRP状态价值函数区别在于与动作有关，与其他状态无关</h3>
</br>
<img width="600" height="152" alt="image" src="https://github.com/user-attachments/assets/95ecb504-8398-4f91-8e07-b49ab48c9440" />
</br>
所有奖励的衰减之和称为回报Gt（Return）
</br>
<img width="496" height="83" alt="image" src="https://github.com/user-attachments/assets/936ad565-6968-4de8-bbc6-69f73e34740d" />
长得和MRP里的状态价值函数差不多
关于<img width="148" height="37" alt="image" src="https://github.com/user-attachments/assets/72463b3e-dcc4-461d-aac8-51762be867fe" />可以展开为<img width="353" height="142" alt="image" src="https://github.com/user-attachments/assets/972e41e6-22e3-4944-a56a-5d8aaf6bd190" />
而那个期望E可以表达为从该状态出发去往各个状态对应价值与概率之和（期望本身就是概率乘以获得值）
</br>
#<h3 align="left">3.3.3动作价值函数</h3>
</br>
<img width="601" height="466" alt="image" src="https://github.com/user-attachments/assets/6c84a8f1-ff85-4374-b997-6eedf541b199" />
</br>
3.3.3动作价值函数总结
</br>
动作价值函数：
</br>
<img width="321" height="48" alt="image" src="https://github.com/user-attachments/assets/36bda09a-b7e4-49c1-8237-0d3f2a472aea" />
</br>
解释为：在状态s下，执行动作a下获得回报G的期望值
</br>
状态价值函数与动作价值函数的关系：
</br>
<img width="256" height="53" alt="image" src="https://github.com/user-attachments/assets/67d5ca1e-06a8-422d-8d20-1727fe035300" />
</br>
</b>状态价值函数V等于策略pai所有概率乘以对应获得的Q之和</b>
</br>
动作价值函数展开
</br>
<img width="411" height="58" alt="image" src="https://github.com/user-attachments/assets/e7528c35-a9f9-414e-9a4c-3913427f044d" />
</br>
</b>动作价值函数展开为：在使用策略pai，状态s下采取动作a的价值等于此时此刻即时奖励r（s,a）加上经过衰减后的所有可能从本状态s出发的下一个状态转移概率与相应价值的乘积</b>
</br>
#<h3 align="left">3.3.4贝尔曼期望方程</h3>
贝尔曼期望方程包括MDP下的V状态价值方程与MDP下的Q动作价值方程
</br>
<img width="566" height="210" alt="image" src="https://github.com/user-attachments/assets/bff002dc-bd4e-4bab-876c-f5541444f506" />
</br>
解释：这里推导主要就是用v=pai x Q 把Q展开带入
</br>
#<h4 align="left">一个例子</h4>
<img width="637" height="415" alt="image" src="https://github.com/user-attachments/assets/951fb670-5be3-4905-9985-c643bdbca28b" />
</br>
<img width="1166" height="755" alt="image" src="https://github.com/user-attachments/assets/6cd40ba3-be9c-459e-9f1f-2ab315c2d43d" />
</br>
<img width="860" height="482" alt="image" src="https://github.com/user-attachments/assets/71079c7e-60f4-4996-b28c-2305c22cb12c" />
</br>
【个人理解】首先在MRP的价值方程里面是没有动作概率这一说的，所以我么们可以把动作价值方程改造为状态价值方程，具体如何人做呢，首先改造r(s,a)那就是把某个s的动作概率乘以对应奖励那么就得出了一个新的r，P(s'|s,a)也是同理，后续再利用MRP里面的：
</br>
<img width="337" height="71" alt="image" src="https://github.com/user-attachments/assets/d3cc0cbd-6bfd-45fc-a50f-560b4f2bd578" />
</br>
联立多个方程组解出每个V(s)接着代入Q的方程就好了
#<h2 align="left">M3.4蒙特卡洛方法</h2>
</br>
<img width="395" height="352" alt="image" src="https://github.com/user-attachments/assets/ddf340bb-e8ba-4e6c-9c00-853f7e3bf6c5" />
</br>
总结：蒙特卡洛方法就是用概率统计来估计数值，首先来回顾下：<img width="1462" height="505" alt="image" src="https://github.com/user-attachments/assets/536c770d-bb83-4c40-b0a9-55fff4ee8bbb" />
</br>
【重点】所以MDP中的蒙特卡洛方法就可以理解为：从s出发，采样n个回报G，求和，然后再除以n就代表状态价值<img width="381" height="80" alt="image" src="https://github.com/user-attachments/assets/2a3bcfa2-ffea-4837-b670-3e976558fc08" />
【G为累计回报！！！！！！！一整条序列叠加】
</br>
【重点】我们介绍的蒙特卡洛价值估计方法会在该状态每一次出现时计算它的回报。还有一种选择是一条序列只计算一次回报，也就是这条序列第一次出现该状态时计算后面的累积奖励，而后面再次出现该状态时，该状态就被忽略了。
</br>
【重点-第一种计算方法，总回报除以计数器，利用大数定律】
<img width="667" height="280" alt="image" src="https://github.com/user-attachments/assets/d193a3e8-d51d-4566-ba4b-44289a2e85ca" />
</br>
【重点-第二种方法：增量式，前一轮的结果加上本轮的平均到n轮，G-V(s)为本轮的实际回报】
【G为累计回报！！！！！！！一整条序列叠加】
<img width="576" height="122" alt="image" src="https://github.com/user-attachments/assets/ccc3281d-9490-43f0-a955-f7554ed85067" />
</br>
#<h2 align="left">M3.5占用度量</h2>
<img width="679" height="168" alt="image" src="https://github.com/user-attachments/assets/b49b4604-f6c8-4a09-8407-40739bb1ed65" />
</br>
<img width="637" height="415" alt="image" src="https://github.com/user-attachments/assets/1e80d1ca-9071-450e-b183-7c40f9ab739d" />
</br>
【解释】如图如果我们选择了一个策略为尽快到达终点s5，那么显然智能体不会从s3先去s4再去s5，即使从s4到s5会有很大的奖励10，所以因为得不到10，所以agent在s3的概率比较小，所以引出不同策略下，agent访问›状态的概率分布不同  
</br>
<img width="1230" height="1064" alt="image" src="https://github.com/user-attachments/assets/363695e5-1cdb-45d1-9a7b-a1e46e4d22e6" />
</br>
<img width="1224" height="516" alt="image" src="https://github.com/user-attachments/assets/f8253d5e-62f2-4c27-8260-26b54e3ba466" />
</br>
#<h2 align="left">M3.6最优策略</h2>
1.首先V *为 至少存在一个策略比其他所有策略都好或者至少存在一个策略不差于其他所有策略，这个策略就是最优策略（optimal policy）
</br>
<img width="1080" height="174" alt="image" src="https://github.com/user-attachments/assets/2c288fd1-9cbf-4985-92cf-faf2a3629503" />
</br>
2.同理，Q *如下
</br>
<img width="988" height="166" alt="image" src="https://github.com/user-attachments/assets/2a3cf41b-1d17-4130-a0b0-216fea74ef8a" />
</br>
3.因为Q的公式为：<img width="648" height="86" alt="image" src="https://github.com/user-attachments/assets/614c0a69-3158-4f4c-832d-94d9c56d2bb6" />，所以要想叫Q最大，那么后续的V‘必然都要为最大，所以V *与Q *关系如下：
</br>
<img width="1234" height="484" alt="image" src="https://github.com/user-attachments/assets/1e1ba2cf-91a8-4f2f-adea-b62a7ab0cfb8" />
///////以上有些不明白
#<h1 align="left">M4-动态规划强化学习算法</h1>
#<h2 align="left">M4.1简介概述</h2>
1.基于动态规划的强化学习算法主要有两种：策略迭代与价值迭代；
</br>
2.策略迭代分为两步：策略评估与策略提升
</br>
3.环境准备
<img width="793" height="732" alt="image" src="https://github.com/user-attachments/assets/4d306e21-1809-4cf4-82bb-7ec6113fc184" />
</br>
<img width="1294" height="166" alt="image" src="https://github.com/user-attachments/assets/957d92b5-f79a-4185-a21e-38f5724dfbfa" />
</br>
一个坐标一个状态 故共有行 * 列个状态s，每个状态下有4个动作
<img width="1414" height="1134" alt="image" src="https://github.com/user-attachments/assets/bb38e85c-f8fe-4553-9853-952478b0f936" />
</br>
如图所示，留在原地了，所以他的状态都是乘以self与本轮的i，j
</br>
<img width="1526" height="998" alt="image" src="https://github.com/user-attachments/assets/a4fda9d0-bb39-4b23-8dce-d561db3889fa" />
</br>
状态编号如何编号的？
<img width="1554" height="1192" alt="image" src="https://github.com/user-attachments/assets/5a01cebd-30a6-41ba-aecd-50f7c1c2f347" />
#<h2 align="left">M4.3策略迭代</h2>
#<h3 align="left">M4.3.1策略评估</h3>
<img width="1392" height="452" alt="image" src="https://github.com/user-attachments/assets/6c449a82-d5f4-4779-bda4-9c0e21634cd0" />
</br>
<img width="636" height="511" alt="image" src="https://github.com/user-attachments/assets/d897ea4f-062e-4d24-8bed-60a29c7dd8c5" />
</br>
<img width="707" height="232" alt="image" src="https://github.com/user-attachments/assets/5bd5d5ac-e200-41ff-beaa-422fabdc0a45" />
总而言之，策略评估的过程为首先随便猜一个初始状态，然后用它算V1然后用V1算V2，最终这个数值会稳定到真正准确的分数<img width="25" height="26" alt="image" src="https://github.com/user-attachments/assets/86963d7f-05f0-435e-8906-a241f80b19ca" />如果这一轮算出的分数和上一轮差值较小那么结束
显而易见，在计算本轮所有“路口”的v的时候要用到上一轮老的v，因为用了新的
#<h3 align="left">M4.3.2策略提升</h3>
<img width="1410" height="598" alt="image" src="https://github.com/user-attachments/assets/378dd8eb-c7d6-4b02-a633-a051d4d8b0ad" />
</br>
策略提升个人理解
1.<img width="71" height="38" alt="image" src="https://github.com/user-attachments/assets/0bf1bd67-1bc1-456f-bc91-727abce3c456" />：如果继续坚持老的策略pai，未来平均得分
<img width="97" height="42" alt="image" src="https://github.com/user-attachments/assets/c0411afc-ebee-4c1e-9278-69cbd890df4b" />：在某一步试了新的动作a之后又回到策略pai得到的得分
</br>
2.策略提升定理
若出现一个新的动作a（新的策略）使得<img width="235" height="41" alt="image" src="https://github.com/user-attachments/assets/bcb741ad-c7bc-4226-bcdd-d90785e695bb" />，那么：
</br>
<img width="297" height="102" alt="image" src="https://github.com/user-attachments/assets/c781c204-75c4-4efc-9eb6-8659c7db90f2" />
</br>
3.贪心算法：如何找到新策略
</br>
<img width="966" height="155" alt="image" src="https://github.com/user-attachments/assets/8b04e50c-e71f-4673-8aa4-7089957ea19f" />
</br>
简而言之就是遍历每一个动作a寻找到最大的那个，采纳为新的策略
</br>
<img width="977" height="102" alt="image" src="https://github.com/user-attachments/assets/7518918d-271d-47fa-9522-18d791f0e80c" />
#<h3 align="left">M4.3.3策略迭代算法</h3>
1.策略迭代算法的过程：评估-提升-评估-提升。。。
2.策略迭代算法的基本流程：
</br>
<img width="924" height="676" alt="image" src="https://github.com/user-attachments/assets/356dd00b-cad0-4649-b76a-6f932573a38d" />
</br>
解释：
</br>
【策略评估阶段】</br>
a.首先令delta=0，当delta大于某个阈值（未收敛）的时候进入策略评估阶段</br>
b.接着遍历全部状态并且赋值给v</br>
c.接着利用这个状态值和贝尔曼方程计算后续其他所有状态的值</br>
d.取出delta与当前状态和其他状态差值中最大的那个，完成收敛</br>
【策略提升阶段】</br>
利用评估阶段得出v结果，遍历所有动作a，找到max那个作为新的策略。</br>
3.策略迭代算法实例演示</br>
<img width="1554" height="1304" alt="image" src="https://github.com/user-attachments/assets/4add159f-b37f-4aa4-b2ec-cb539d08a209" />
</br>
<img width="702" height="476" alt="image" src="https://github.com/user-attachments/assets/59ad2a20-66dc-4565-a72f-6077b4e3a4e8" />
</br>
a.策略评估和策略提升在policy_iteration中交替执行，他们通过如下方式联系起来：因为在策略提升中的贪心寻找最大的Q需要用到各个状态的v，所以这里就用到了策略评估当中算出来的在采取不同策略pai下不同状态值
</br>
b.如何实现计算的？
</br>
<img width="593" height="134" alt="image" src="https://github.com/user-attachments/assets/9138a5f9-683a-4cab-87a6-b813b72d9482" />
</br>
qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
</br>
这里公式原型实际上为原始的状态价值方程也为确定某策略情况下的q<img width="382" height="37" alt="image" src="https://github.com/user-attachments/assets/4d23c91c-5b00-4608-bac6-437d51d1a6c0" />p为例子中，奖励r也与状态转移概率p相关所以要乘到前面
</br>
qsa_list.append(self.pi[s][a] * qsa)
</br>
new_v[s] = sum(qsa_list)
</br>
这里表示<img width="460" height="112" alt="image" src="https://github.com/user-attachments/assets/15a805ac-e0ee-4f67-8c05-0a543621c207" />
</br>
确定某策略下的动作价值函数和状态价值函数一样
</br>
#<h2 align="left">M4.4价值迭代算法</h2>
</br>
与上面不同，上面的策略迭代为评估-迭代-评估。。。。。。
但是这里在策略评估中直接根据更新的价值进行策略提升
价值迭代（Value Iteration） 的核心思想就是：把策略评估的过程“砍”到只剩一步，然后直接取最大值进行更新。这就好比你不需要把迷宫的每一条死胡同都走到头才能决定哪条路更好，你只要往前看一步，选那个预期得分最高的就行了。价值迭代（Value Iteration） 的核心思想就是：把策略评估的过程“砍”到只剩一步，然后直接取最大值进行更新。这就好比你不需要把迷宫的每一条死胡同都走到头才能决定哪条路更好，你只要往前看一步，选那个预期得分最高的就行了。
</br>
<img width="1800" height="1206" alt="image" src="https://github.com/user-attachments/assets/0ea1c4d7-fbb7-41ad-b0f5-e3eadb6e045b" />
</br>
qsa_list.append(qsa)  # 这一行和下一行代码是价值迭代和策略迭代的主要区别
</br>
new_v[s] = max(qsa_list)
</br>
这个Q直接被当作普通状态价值v来计算了
</br>
#<h2 align="left">M4.5动态规划小结</h2>
1.策略迭代算法用的是贝尔曼期望方程</br>
<img width="442" height="87" alt="image" src="https://github.com/user-attachments/assets/c1b981fe-54d9-4cc9-9a70-50ae28533a63" /></br>
他的过程为：评估-更新-评估-更新。。。。。。直到评估函数收敛，然后利用策略提升定理选出max的
上面的贝尔曼期望方程可以理解为，尝试s下所有动作用其概率乘以相对应的q值，再把所有动作相加，最后找到max那个
利用上一轮中算出来的下一个状态值评估本轮本状态的价值
</br>
<img width="903" height="352" alt="image" src="https://github.com/user-attachments/assets/4c3752fd-2f75-4155-b6e4-6a6a4c7bdc45" />
</br>
<img width="963" height="397" alt="image" src="https://github.com/user-attachments/assets/7a161c67-c459-4e9a-b765-8a76bc4beb06" />
</br>
2.价值迭代算法用的是贝尔曼最优方程
</br>
<img width="391" height="54" alt="image" src="https://github.com/user-attachments/assets/2406e2c3-6f15-41f7-ac84-988f1945d697" />
</br>
仅维护价值V(s)，无显式策略，每次仅仅更新一轮V，直接更新策略为max那个
#<h1 align="left">M5时序差分算法</h1>
#<h2 align="left">M5.1简介</h2>
其马尔可夫决策过程的状态转移概率是无法写出来的，也就无法直接进行动态规划。在这种情况下，智能体只能和环境进行交互，通过采样到的数据来学习，这类学习方法统称为无模型的强化学习
</br>
无模型的强化学习算法不需要事先知道环境的奖励函数和状态转移函数，而是直接使用和环境交互的过程中采样到的数据来学习，这使得它可以被应用到一些简单的实际场景中。本章将要讲解无模型的强化学习中的两大经典算法：Sarsa 和 Q-learning，它们都是基于时序差分（temporal difference，TD）的强化学习算法。
</br>
在线策略学习：使用在当前策略下采样得到的样本进行学习，一旦策略被更新，当前的样本就被放弃了
</br>
离线策略学习：线策略学习使用经验回放池将之前采样得到的样本收集起来再次利用，往往能够更好地利用历史数据，并具有更小的样本复杂度
#<h2 align="left">M5.2时序差分基本方法</h2>
1.回顾下蒙特卡洛对于价值函数的增量式更新
</br>
<img width="326" height="70" alt="image" src="https://github.com/user-attachments/assets/5027dec7-5c9b-4740-9809-0de83c29f647" />
</br>
2.时序差分方法只需要当前步结束即可进行计算。具体来说，时序差分算法用当前获得的奖励加上下一个状态的价值估计来作为在当前状态会获得的回报，他的更新方式如下：
</br>
<img width="395" height="58" alt="image" src="https://github.com/user-attachments/assets/af4755ac-7184-4eee-8957-0620e4d66f82" />
</br>
【个人理解上述公式】
2.1</br>
<img width="911" height="317" alt="image" src="https://github.com/user-attachments/assets/4a87e8b0-28ea-465a-8ced-e12e45ac62e1" />
</br>
2.2</br>
我们可以把前面的<img width="122" height="33" alt="image" src="https://github.com/user-attachments/assets/1f34fc21-bb22-4d58-bf2a-a10a4fdd3efa" />
认为是根据现实环境，我们结合实际估计出来新的v，减去我们以前估计的v，那么就得到了一个误差，这个误差用来调整策略，这正好也对应了时序差分算法在样本与环境中学习。
</br>
<img width="875" height="292" alt="image" src="https://github.com/user-attachments/assets/764df946-ede1-49ce-bb5d-373a166bc7bb" />
</br>
新的估计值 = 旧的估计值 + 学习率 × ( 误差 )
</br>
#<h2 align="left">M5.3Sarsa</h2>
1.在时序差分的情境下，我们不清楚数据分布，奖励分布，价值分布，即我们不知道奖励函数和状态转移函数，那么该如何进行策略提升，那么我们可以直接利用Q的时序差分，即老的动作价值Q+插值，更新为新的Q，然后用贪婪算法选取max作为新的动作策略
</br>
<img width="527" height="58" alt="image" src="https://github.com/user-attachments/assets/f870a1e8-94ab-4a4d-9eb6-e91662b9f77c" />
</br>
2.如果在策略提升中一直根据贪婪算法得到一个确定性策略，可能会导致某些状态动作对永远没有在序列中出现
</br>
<img width="535" height="95" alt="image" src="https://github.com/user-attachments/assets/39bbf6fb-7d1f-44fc-88fe-88b2a4feb886" />
</br>
3.Sarsa算法流程
</br>
<img width="250" height="50" alt="image" src="https://github.com/user-attachments/assets/0683e374-9ff6-40a7-b70c-964a9369639a" />为实际的本轮Q值，因为用到了实际的r以及预估的下一个状态，更新完毕之后，把猜测的动作与价值更新为本轮的
</br>
<img width="587" height="87" alt="image" src="https://github.com/user-attachments/assets/0ea3cca7-00d4-491a-982b-20b04a86185c" />
</br>
<img width="571" height="412" alt="image" src="https://github.com/user-attachments/assets/09170ced-9e5d-4a33-a18f-de108885c429" />
</br>
简单来讲这个算法流程为：首先进入大循环（也就是Episode 循环）
</br>
然后根据 epsilon-greedy 选择当前动作a
</br>
接着进入时间步循环，根据环境反馈也就是现实中的s'等得到实际的Q，这里要用的e-greedy策略，可能选择max，也可能随机选择一个接着用差值更新Q
它算出“实际”和“预期”的差距（TD 误差），然后用学习率alpha修正自己脑子里的 Q 表格。
</br>
<img width="162" height="27" alt="image" src="https://github.com/user-attachments/assets/bf9a909d-71bf-46da-9700-315ae549d5f3" />把刚才的“新位置”和“预判的下一步动作”，变成当下的“老位置”和“老动作”。
</br>
内层循环每走一步，都会在 Q 表格上写下新心得；当这一局结束（外层循环进入下一局）时，AI 会带着上一局写满心得的 Q 表格重新站在起点！
外围的循环Episodes最终走向收敛！！！！！！
</br>
#<h2 align="left">Sarsa总结</h2>
1.输入输出
</br>
<img width="787" height="567" alt="image" src="https://github.com/user-attachments/assets/154d53fd-39c3-4b40-841b-d5f21e3ff266" />
</br>
2.算法流程
</br>
<img width="790" height="477" alt="image" src="https://github.com/user-attachments/assets/f59df949-38c9-4fae-8c14-ef81f3384985" />
</br>
#<h2 align="left">M5.4多步Sarsa算法</h2>
1.蒙特卡洛方法无偏，因为他不仅仅聚焦下一个状态与下一个奖励，但是因为他关注的太多了，导致就方差大
</br>
2.Sarsa算法小方差，但是有偏，仅关注下一步
</br>
3.解决：多步Sarsa
</br>
<img width="587" height="291" alt="image" src="https://github.com/user-attachments/assets/920d0d06-dd44-4aaa-9c92-822247aafa21" />
</br>
#<h2 align="left">多步Sarsa总结</h2>
</br>
<img width="822" height="397" alt="image" src="https://github.com/user-attachments/assets/50014ade-c0a9-4a2e-921c-a50030fa2d9f" />
</br>
往后看n步！计算差值！然后注意折扣因子的指数
</br>
<img width="786" height="672" alt="image" src="https://github.com/user-attachments/assets/54f1f649-c778-4701-a1cc-8536f83ce885" />
</br>
#<h2 align="left">M5.5Q-learning</h2>
</br>
1.Q-learning的时序查分更新方式为
</br>
<img width="473" height="44" alt="image" src="https://github.com/user-attachments/assets/03d3ecf1-b7f2-4595-a9c7-1f601302f1e2" />
</br>


2.【这种方式个人理解】
</br>
<img width="115" height="32" alt="image" src="https://github.com/user-attachments/assets/8aa1275b-0f50-483e-a1c4-6926beb64eed" />
</br>
这里为Q-learning与Sarsa的区分点，Sarsa中根据实际预估的下一个状态的动作a为利用e-greedy策略选取的动作a，而Q-learning这里为遍历动作a
算法直接把st+1这一行在Q表格中对应的所有动作价值全部调出来，进行一次遍历求最大值的操作
不过，当下动作a还是要利用e-greedy来选取
完成这一切之后和Sarsa类似，不断更新，需要注意的是，在末尾并没有更新预估的动作a为下一轮的a，这是因为，a都是遍历max
</br>


3.Q-learning为离线策略 off-policy
</br>
在线策略（on-policy）算法表示行为策略和目标策略是同一个策略；
而离线策略（off-policy）算法表示行为策略和目标策略不是同一个策略。
</br>
Q-learning用于更新公式的策略max和用于实际行动的离线策略e-greedy不是同一个，所以它被称为离线算法。
解释：他用于下轮预估的a用了max，而他实际行动（即在内循环中当下的动作a用的是e-greedy）
</br>
Sarsa论更新还是行动都绑定同一个e-greedy采样，所以叫在线策略。观察这里<img width="441" height="103" alt="image" src="https://github.com/user-attachments/assets/b0a8ec79-7cc2-45ec-8324-daa0ce193a89" />可以得到Sarsa用于下一轮预估的a用了e-greedy，而他实际行动，在内循环中当下的动作，也用了这个e-greedy
</br>
4.Q-learning的算法流程
</br>
<img width="566" height="422" alt="image" src="https://github.com/user-attachments/assets/cf7d3ff5-6a53-4890-95c0-8ffcaa55260e" />
</br>
<img width="1242" height="558" alt="image" src="https://github.com/user-attachments/assets/7c0b8cec-8918-4533-989e-fe29444dfbcc" />
</br>
#<h2 align="left">Q-learning总结</h2>
</br>
<img width="115" height="32" alt="image" src="https://github.com/user-attachments/assets/8aa1275b-0f50-483e-a1c4-6926beb64eed" />
</br>
这里为Q-learning与Sarsa的区分点，Sarsa中根据实际预估的下一个状态的动作a为利用e-greedy策略选取的动作a，而Q-learning这里为遍历动作a
算法直接把st+1这一行在Q表格中对应的所有动作价值全部调出来，进行一次遍历求最大值的操作
不过，当下动作a还是要利用e-greedy来选取
完成这一切之后和Sarsa类似，不断更新，需要注意的是，在末尾并没有更新预估的动作a为下一轮的a，这是因为，a都是遍历max
</br>
<img width="765" height="525" alt="image" src="https://github.com/user-attachments/assets/62780fbf-30c8-45ab-87d0-f86d8573c1de" />
</br>
Q-learning为离线策略
</br>
<img width="418" height="193" alt="image" src="https://github.com/user-attachments/assets/d47b61fe-e548-4e32-ae7b-3db30905dbe8" />

#<h2 align="left">时序差分Part总结</h2>
</br>
1.Sara算法
</br>
<img width="450" height="42" alt="image" src="https://github.com/user-attachments/assets/e53c678a-27c9-4377-b4a6-e615efc1b984" />
</br>
2.多步Sarsa
</br>
<img width="577" height="52" alt="image" src="https://github.com/user-attachments/assets/667f5a1b-1524-4775-b462-f9c881986eac" />
</br>
往后看n步！计算差值！然后注意折扣因子的指数
</br>
3.Q-learning
</br>
<img width="475" height="58" alt="image" src="https://github.com/user-attachments/assets/ba1d84de-8756-45f5-b260-c71c70c8e32e" />
</br>
Sarsa中根据实际预估的下一个状态的动作a为利用e-greedy策略选取的动作a，而Q-learning这里为遍历动作a
算法直接把st+1这一行在Q表格中对应的所有动作价值全部调出来，进行一次遍历求最大值的操作
Q-learning用于更新公式的策略max和用于实际行动的离线策略e-greedy不是同一个，所以它被称为离线算法。
解释：他用于下轮预估的a用了max，而他实际行动（即在内循环中当下的动作a用的是e-greedy）
</br>
#<h1 align="left">M6 Dyna-Q算法</h1>
#<h2 align="left">M6.1 概述</h2>
</br>
强化学习算法分为两种 无模型 和 有模型
</br>
1.无模型：Sarsa,Q-Learning
</br>
2.有模型：策略迭代，价值迭代，Dyna-Q 算法；特别的Dyna-Q 算法的环境模型是通过采样数据估计得到的
</br>
#<h2 align="left">M6.2Dyna-Q</h2>
</br>
Dyna-Q 使用一种叫做 Q-planning 的方法来基于模型生成一些模拟数据，然后用模拟数据和真实数据一起改进策略。
先选取经验数据里面访问过的状态s，采取曾经在该状态s下执行过的动作a，通过学到的环境模型模拟出来后续状态
s'与奖励r，并根据这个模拟数据<s,a,r,s'>，用 Q-learning 的更新方式来更新动作价值函数。
</br>
<img width="592" height="668" alt="image" src="https://github.com/user-attachments/assets/7111e956-486f-4fa0-be6a-026a654d17fa" />
</br>
【个人理解】Dyna-Q 就像边试错、边画地图、还爱在脑子里复盘
</br>
1.准备阶段初始化一个Q表格，一个环境模型M，前者记录在哪个状态下做哪个动作得分，后者记录地图规律
</br>
2.根据e-greedy选择当下动作a，与环境交互得到r，s'---->Q-learning-------->把刚才摸索出的环境规律M(s,a)<----r,s'记下来
</br>
3.【Q-Planning】复盘M，找一个过去的状态和动作a，看记录下的rm，s'm是啥再利用Q-learning更新Q表格
</br>
【输出】
算法运行结束（跑完E个序列）后，整体输出的是一个收敛的Q表格（最优动作价值函数Q(s,a)），以及顺带学习到的一个环境模型M。
</br>
#<h1 align="left">M7 DQN算法(deep Q network)</h1>
#<h2 align="left">M7.1 简介</h2>
#<h2 align="left">DQN为离线算法</h2>
</br>
DQN算法是深度强化学习的基础。在Q-Learning中的Q为一张表格，矩阵，遍历某一行，找到max那个，这只适合在离散且数据量比较小的情况下适用。
</br>
状态动作都连续的情况下，s-a状态-动作对为无限个，无法用表格记录，需要用函数拟合的方法来估计Q值，即将这个复杂的Q值表格视作数据，使用一个参数化的函数<img width="23" height="26" alt="image" src="https://github.com/user-attachments/assets/17d2a3bf-aee7-4096-bd96-ab2a0400ac27" />来拟合这些数据。很显然，这种函数拟合的方法存在一定的精度损失，因此被称为近似方法。
</br>
#<h2 align="left">M7.2 DQN(deep Q network)</h2>
</br>
用神经网络来表示以前的Q网络
</br>
1.如果输入的状态s和a为连续的，输出的为一个标量，表示在状态s下采取动作a能获得的价值。
【解释】面对无数种可能的连续动作，我们只能把状态和某个具体动作一起当成输入，让网络针对这一种情况，给出一个孤零零的标量。
</br>
2.如果输入的为离散数据，除了可以采取动作连续情况下的做法，我们还可以只将状态s输入到神经网络中，使其同时输出每一个动作Q的值。
【解释】输入为n个动作，输出为一个n个Q值的向量
</br>
3.拟合的函数为w，每一个状态s下所有可能动作a的值我们都能表示为<img width="78" height="31" alt="image" src="https://github.com/user-attachments/assets/9e32d8ee-80fd-4d13-9ea2-8380cb15630e" />
</br>
4.Q网络结构图
</br>
<img width="617" height="542" alt="image" src="https://github.com/user-attachments/assets/0d0509a7-807a-4bbb-9e39-effbf02bd65e" />
</br>
5.损失函数
</br>
<img width="915" height="388" alt="image" src="https://github.com/user-attachments/assets/c8a79a96-811b-4485-add8-91eb9dc758d7" />
</br>
#<h3 align="left">M7.2.1 经验回放</h3>
DQN 算法采用了经验回放（experience replay）方法，具体做法为维护一个回放缓冲区，将每次从环境中采样得到的四元组数据（状态、动作、奖励、下一状态）存储到回放缓冲区中，训练 Q 网络的时候再从回放缓冲区中随机采样若干数据来进行训练。
</br>
作用：采用经验回放可以打破样本之间的相关性，让其满足独立假设；提高样本效率。每一个样本可以被使用多次。
</br>
#<h3 align="left">M7.2.2 目标网络</h3>
</br>
按照前面讲的Qnet会导致训练不稳定，因为其核心目的是叫预测值Qw去逼近目标值<img width="248" height="28" alt="image" src="https://github.com/user-attachments/assets/2662efc0-2023-4726-933a-05af5569db4f" />
</br>
预测值和目标值里，用的都是同一个神经网络QW,但在没有目标网络的 DQN 里，每次智能体走了一步、更新了自己的大脑参数w之后，用来计算目标位置的公式也跟着变了
</br>
1.Qw
</br>
用于计算原来的损失函数中的<img width="78" height="23" alt="image" src="https://github.com/user-attachments/assets/767f8390-def2-4253-83f2-d459e1a4e685" />项，并且使用正常梯度下降方法来进行更新。负责输出当前的预测值
</br>
2.Qw-
</br>
专门用来计算那个目标值
</br>
为了让更新目标更稳定，目标网络并不会每一步都更新。具体而言，目标网络使用训练网络的一套较旧的参数，训练网络Qw-在训练中的每一步都会更新，而目标网络的参数每隔C步才会与训练网络同步一次，即w- <-w。这样做使得目标网络相对于训练网络更加稳定。
</br>
DQN算法流程
</br>
<img width="631" height="673" alt="image" src="https://github.com/user-attachments/assets/fdc52177-a179-46c9-8bd4-39ed75a28698" />
</br>
#<h2 align="left">Q-learning、DQN 及 DQN 改进算法都是基于价值（value-based）的方法，他们都是学习值函数，然后从中直接导出一个策略，没有一个显式维护策略的过程
其中 Q-learning 是处理有限状态的算法，而 DQN 可以用来解决连续状态的问题。，这是因为前者处理离散，DQN是网络结构，大批量且连续</h2>
</br>
#<h1 align="left">M8 DQN改进算法</h1>
#<h2 align="left">M8.1 简介</h2>
DQN之后，有很多改进的算法，Double DQN 和 Dueling DQN
#<h2 align="left">M8.2 Double DQN</h2>
<img width="922" height="324" alt="image" src="https://github.com/user-attachments/assets/a25dcdaf-57dc-43d1-ba27-4197533ccc83" />
</br>
【解释】这里的优化目标TD可以理解为结合现实中的r与s算出来的目标值，用他减去模型预测出来的Q用来调整参数w
</br>
1.Qw网络和Qw-网络
</br>
Qw网络：原本的训练网络用于计算损失函数里面的Qw(s,a)项
Qw-网络:目标网络用于计算原来损失函数里的Qw-(s,a)
</br>
2.在普通DQN里面采用第二套独立的Qw-目标网络并且他的权重参数每隔一段时间才更新一次由于 TD 误差目标本身就包含神经网络的输出，因此在更新网络参数的同时目标也在不断地改变，这非常容易造成神经网络训练的不稳定性。为了解决这一问题，DQN 便使用了目标网络
</br>
<img width="915" height="388" alt="image" src="https://github.com/user-attachments/assets/c8a79a96-811b-4485-add8-91eb9dc758d7" />
</br>
3.普通DQN里面结合现实和Q-learning的误差目标为<img width="200" height="38" alt="image" src="https://github.com/user-attachments/assets/bfb2aeb5-ae23-4098-8481-c086ee07a190" />，先去取出来那一行mxQ的动作a然后再计算新的Q所以这里可以简化为
</br>
<img width="272" height="57" alt="image" src="https://github.com/user-attachments/assets/5ba1f321-4227-4f3c-abbd-d22ab100c303" />
</br>
但是这种方式容易造成如果Qw-正向负向误差累积采用了同一个网络那么会产生大的误差所以把里面取maxQ的动作a的网络换为原本的网络Qw
</br>
<img width="328" height="86" alt="image" src="https://github.com/user-attachments/assets/cec9716f-a3d8-436d-a93b-c58dac774be6" />
</br>
#<h2 align="left">M8.3 Dueling DQN</h2>
</br>
1.优势函数A
</br>
A:将动作价值函数Q减去状态价值V，即<img width="217" height="32" alt="image" src="https://github.com/user-attachments/assets/23cb1182-8aac-43d9-addb-7d7e95df523f" />在同一个状态下，所有动作的优势值(A)之和为 0
2.Dueling DQN中的Q网络为
</br>
<img width="573" height="280" alt="image" src="https://github.com/user-attachments/assets/a588768e-8cc7-48ad-b7b5-b2802c242485" />
</br>
[解释]
</br>
a.如果 A > 0，说明这个动作比平均水平好；如果 A < 0，说明出了个昏招。
</br>
b.从纯数学公式上看：既然 A = Q - V，那网络输出 V + A 不就等于V + (Q - V) = Q吗？为什么不直接让网络输出Q呢？
</br>
左脑（V分支）： 专门负责评估“当前局面到底有多好”，输出一个标量V(s)。
</br>
右脑（A分支）： 专门负责评估“各个动作相比于大局面的优势”，输出一组向量 A(s, a) 这里的A是神经网络里面结合现在平均水平计算出来的。
</br>
前面说的A(s, a) = Q(s, a) - V(s)，这是一个理论上的定义式。这里的Q和V是理想状态下完美的真实价值。
</br>
截图后面公式 Q_{\eta, \alpha, \beta}(s, a) = V_{\eta, \alpha}(s) + A_{\eta, \beta}(s, a)，这是神经网络的计算式。这里的V和A是网络两个分支瞎猜出来（并不断学习优化）的值，相加后得到的Q是网络的预测输出值。所以，它们在概念上是指向同一个东西，但在实际网络运行中，并没有用一个现成的 Q 去计算 A。
</br>
<img width="573" height="536" alt="image" src="https://github.com/user-attachments/assets/f2875052-4fe4-40a6-a0ca-923a50f549b8" />
</br>
c.为啥要这样子设计呢？
</br>
如果你用传统 DQN，它必须给每一个动作单独学习一个很低的 Q值。按喇叭的Q值要学很久才能降下来，换频道的Q值也要学很久才能降下来……它没有“整体局势很糟”这个概念。
</br>
但是Dueling DQN中V分支迅速察觉到现在情况非常非常危险所有的得分都非常非常低所以只需要去聚焦A的微小差异“向左转虽然也会撞，但只蹭到边，优势A稍微高一点；按喇叭必死无疑，优势A最低。”
</br>
d.
</br>
<img width="598" height="460" alt="image" src="https://github.com/user-attachments/assets/e456ad69-40fd-41a3-ad3a-289c8b4b4109" />
</br>
假设某个状态下动作的真实 $Q$ 值应该是 $100$。理想情况： 网络乖乖学到了局势分 $V = 80$，动作优势分 $A = 20$。$80 + 20 = 100$。很完美。作妖情况 1： 网络的 $V$ 分支突然抽风，输出了 $V = 1000$。为了强行凑出 $100$ 的结果，$A$ 分支就会被迫输出 $A = -900$。$1000 + (-900) = 100$。作妖情况 2： $V$ 输出 $-500$，$A$ 输出 $600$。$-500 + 600 = 100$。你看，只要我给 $V$ 加上任意一个常数 $C$，再给 $A$ 减去同一个常数 $C$，最终加起来的 $Q$ 值永远不变。这就导致了一个可怕的后果：网络的 $V$ 和 $A$ 就像在玩跷跷板，可以漫天乱飞，互相补偿。 这种没有约束的波动会让神经网络的训练极其不稳定，根本学不到真实的“局势”和“优势”。
</br>
