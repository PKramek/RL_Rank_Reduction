# Low rank aproxximation in Reinforcement Learning problems

This repository contains code used when experimenting with Layer-Selective Rank Reduction (LASER) [1] in the context of a reinforcement learning framework. In the paper, the authors described that using the Low-Rank approximation on the Decision Transformer [2] gives a 3% increase in performance on the Sokoban task. However, there is no discussion of the performance delta in the context of control problems with continuous action and observation spaces.

In the experiments, I only modified the trained policy network $\Pi$. I did not modify the $\Pi$, $Q$ and $V$ networks during training, but this might be an interesting area to explore.

## Experiments results

| layer | target_rank | mean_reward         | std_reward          | mean_reward_delta   | std_reward_delta    |
| ----- | ----------- | ------------------- | ------------------- | ------------------- | ------------------- |
| 4*     | 6**           | 5118.12175909224    | 71.46035253699091   | **53.86377664187421**   | **-412.3522091505959**  |
| 2     | 251         | 5114.784104478223   | 86.10277523500716   | **50.52612202785713**   | **-397.7097864525797**  |
| 0     | 17          | 5016.641569900629   | 638.1735183894699   | -47.6164125497362   | 154.36095670188308  |
| 2     | 201         | 4974.570160872144   | 483.07604632969276  | -89.6878215782217   | -0.7365153578940635 |
| 2     | 226         | 4916.723239957853   | 915.0345070982617   | -147.53474249251212 | 431.22194541067483  |
| 2     | 176         | 3837.6266700963315  | 2201.748937681408   | -1226.631312354034  | 1717.936375993821   |
| 0     | 14          | 2566.1663794380647  | 1895.343299856853   | -2498.0916030123008 | 1411.5307381692662  |
| 0     | 16          | 2473.1641397329313  | 1304.8760833000006  | -2591.093842717434  | 821.0635216124138   |
| 0     | 15          | 2246.250280320954   | 930.9153584173253   | -2818.0077021294114 | 447.10279672973843  |
| 2     | 76          | 1121.2707354053762  | 554.5555776114777   | -3942.987247044989  | 70.74301592389088   |
| 4     | 1           | 909.0357388185206   | 116.4285990547551   | -4155.222243631845  | -367.38396263283175 |
| 0     | 12          | 577.2700479205051   | 442.18362024250075  | -4486.9879345298605 | -41.62894144508607  |
| 0     | 13          | 500.0829672273461   | 701.4660973865282   | -4564.17501522302   | 217.65353569894137  |
| 0     | 11          | 491.02274263816975  | 461.19693847201756  | -4573.235239812196  | -22.615623215569258 |
| 2     | 26          | 455.4488698853404   | 306.20438102297953  | -4608.809112565025  | -177.6081806646073  |
| 4     | 4           | 385.2437464058533   | 412.7181041782176   | -4679.0142360445125 | -71.09445750936925  |
| 0     | 3           | 376.9999116196185   | 174.6825458009774   | -4687.258070830747  | -309.1300158866094  |
| 2     | 1           | 265.27493082078405  | 82.35433943881677   | -4798.9830516295815 | -401.4582222487701  |
| 2     | 101         | 255.0842015600916   | 607.6002810054205   | -4809.1737808902735 | 123.78771931783365  |
| 4     | 2           | 222.07891744944035  | 99.41827067844548   | -4842.179065000925  | -384.39429100914134 |
| 0     | 1           | 208.60767920502752  | 127.17034577158088  | -4855.650303245338  | -356.64221591600597 |
| 0     | 9           | 93.90444925803686   | 222.46139398900968  | -4970.3535331923285 | -261.3511676985771  |
| 4     | 3           | 78.11414303580067   | 126.60957761249166  | -4986.143839414565  | -357.2029840750952  |
| 0     | 8           | 51.37218714037723   | 59.516262161242295  | -5012.885795309989  | -424.2962995263445  |
| 2     | 51          | 46.22472165321437   | 160.42192461721925  | -5018.033260797151  | -323.3906370703676  |
| 0     | 10          | 41.395684187897494  | 142.83687212513982  | -5022.862298262468  | -340.975689562447   |
| 0     | 7           | 17.15845096912002   | 3.9947270745417303  | -5047.099531481245  | -479.8178346130451  |
| 0     | 4           | 1.782204063791196   | 0.632682324608901   | -5062.475778386574  | -483.17987936297794 |
| 0     | 6           | 1.715068179685113   | 0.5450369901328894  | -5062.542914270681  | -483.26752469745395 |
| 0     | 5           | -0.32478306309451   | 1.335810733640218   | -5064.58276551346   | -482.4767509539466  |
| 2     | 151         | -2.4988439644966274 | 0.2174971943622991  | -5066.756826414862  | -483.5950644932245  |
| 4     | 5           | -2.6595342504698785 | 0.24641824376263155 | -5066.917516700835  | -483.5661434438242  |
| 2     | 126         | -3.683106993660331  | 0.1606432542335362  | -5067.941089444026  | -483.6519184333533  |
| 0     | 2           | -9.173358096000916  | 2.0116692365748974  | -5073.431340546366  | -481.8008924510119  |

\* The layer numbers correspond to the indexes in the Sequential layer of the policy network $\Pi$. When the layer number is equal to 4, it means that I modified the head layer $mu$.

\*\*Max rank of weight matricies:
  | layer | max_rank  |
  |-------|-----------|
  |0      | 17        |
  |2      | 256       |
  |4      | 6         |

## Results

Even though the policy network is not overparametrized (i.e. we do not use thousands of neurons in each layer), the rank reduction of the weight matrices increased the performance of the agent by approximately **1%**. Interestingly, when applying low-rank aproximation with the target rank equal to the original rank of the matrix in the last layer (first row in the results table), we also get a performance boost.

## Notes

- As of now, I only performed experiments on a single environment and policy obtained by using a single RL algorithm. To draw any objective conclusions, more experiments need to be performed. The goal of the described experiment was only to test a hypothesis, that the idea described in the LASER paper can be applied to smaller networks, not only Transformers.  
- As a starting point of hyperparameter optimization process, I used hyperparameters from the SAC-CEPO paper [3].

## Bibliography

- [1] - Sharma, P., Ash, J.T. and Misra, D., 2023, October. The Truth Is In There: Improving Reasoning with Layer-Selective Rank Reduction. In The Twelfth International Conference on Learning Representations. <https://arxiv.org/abs/2312.13558>
- [2] - Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A. and Mordatch, I., 2021. Decision transformer: Reinforcement learning via sequence modeling. Advances in neural information processing systems, 34, pp.15084-15097. <https://arxiv.org/abs/2106.01345>
- [3] - Shi, Z. and Singh, S.P., 2021. Soft actor-critic with cross-entropy policy optimization. arXiv preprint arXiv:2112.11115. <https://arxiv.org/abs/2112.11115>