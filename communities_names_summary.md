# Communities and Crime 数据集说明总结

## 1. 数据集基本信息

- 数据集名称：`Communities and Crime`
- 数据来源整合自三部分：
  - `1990 US Census` 社会经济与人口普查数据
  - `1990 US LEMAS survey` 执法机构管理与行政统计数据
  - `1995 FBI UCR` 犯罪统计数据
- 数据任务类型：`Regression`
- 样本量：`1994` 个社区
- 属性数：`128`
  - `122` 个预测变量
  - `5` 个非预测变量
  - `1` 个目标变量
- 变量类型：以 `real-valued numeric` 为主，另有一个字符串变量 `communityname`
- 应用领域：社会科学 / 犯罪与社区研究

## 2. 研究目标

该数据集的核心目标是预测社区层面的暴力犯罪水平。被预测的目标变量是：

- `ViolentCrimesPerPop`

它表示社区中每 `10万人` 的暴力犯罪数量，对应的暴力犯罪包括：

- murder
- rape
- robbery
- assault

在数据文件中，该目标变量已经被归一化到 `0-1` 区间。

## 3. 数据集构成逻辑

这个数据集不是随意拼接的变量集合，而是围绕“哪些社区特征可能和犯罪有关”来组织的。作者说明中提到：

- 变量被挑选进来，是因为它们“有可能与犯罪存在合理联系”
- 数据集非常适合测试变量选择、特征权重学习以及回归预测方法
- 数据既包含社区本身的人口、家庭、收入、住房、教育、就业等特征，也包含部分警务资源和执法配置特征

因此，这个数据集在课程项目中有很好的解释空间，不只是能做预测，还能做回归解释、变量筛选、诊断分析和模型比较。

## 4. 变量分组概览

`communities.names` 中虽然按单变量逐条列出，但从内容上可以将变量分成以下几大类。

### 4.1 标识类 / 非预测变量

这几列主要用于识别样本，不适合直接作为解释变量：

- `state`：州编号，作者建议若使用应视为名义变量
- `county`：县编号，且缺失较多
- `community`：社区编号，且缺失较多
- `communityname`：社区名称，仅供识别
- `fold`：用于非随机 `10-fold cross validation` 的折编号

### 4.2 人口与种族结构变量

这一类描述社区人口规模、年龄结构、城市化水平、种族构成等，例如：

- `population`
- `householdsize`
- `racepctblack`
- `racePctWhite`
- `racePctAsian`
- `racePctHisp`
- `agePct12t21`
- `agePct12t29`
- `agePct16t24`
- `agePct65up`
- `numbUrban`
- `pctUrban`

### 4.3 收入、贫困与教育变量

这一类是解释犯罪率时非常重要的一组社会经济变量，例如：

- `medIncome`
- `medFamInc`
- `perCapInc`
- `whitePerCap`
- `blackPerCap`
- `HispPerCap`
- `NumUnderPov`
- `PctPopUnderPov`
- `PctLess9thGrade`
- `PctNotHSGrad`
- `PctBSorMore`

### 4.4 就业与职业结构变量

这组变量刻画劳动市场状态和职业结构，例如：

- `PctUnemployed`
- `PctEmploy`
- `PctEmplManu`
- `PctEmplProfServ`
- `PctOccupManu`
- `PctOccupMgmtProf`

### 4.5 家庭结构与婚姻变量

这部分变量在该数据集中很重要，而且和目标变量相关性较强，例如：

- `MalePctDivorce`
- `MalePctNevMarr`
- `FemalePctDiv`
- `TotalPctDiv`
- `PersPerFam`
- `PctFam2Par`
- `PctKids2Par`
- `PctYoungKids2Par`
- `PctTeen2Par`
- `PctWorkMomYoungKids`
- `PctWorkMom`
- `NumIlleg`
- `PctIlleg`

### 4.6 移民与语言变量

例如：

- `NumImmig`
- `PctImmigRecent`
- `PctImmigRec5`
- `PctImmigRec8`
- `PctImmigRec10`
- `PctRecentImmig`
- `PctRecImmig5`
- `PctRecImmig8`
- `PctRecImmig10`
- `PctSpeakEnglOnly`
- `PctNotSpeakEnglWell`

### 4.7 住房与居住条件变量

这部分变量非常丰富，可用于刻画社区稳定性、拥挤度、房屋质量与住房价值，例如：

- `PctLargHouseFam`
- `PctLargHouseOccup`
- `PersPerOccupHous`
- `PersPerOwnOccHous`
- `PersPerRentOccHous`
- `PctPersOwnOccup`
- `PctPersDenseHous`
- `PctHousLess3BR`
- `MedNumBR`
- `HousVacant`
- `PctHousOccup`
- `PctHousOwnOcc`
- `PctVacantBoarded`
- `PctVacMore6Mos`
- `MedYrHousBuilt`
- `PctHousNoPhone`
- `PctWOFullPlumb`
- `OwnOccLowQuart`
- `OwnOccMedVal`
- `OwnOccHiQuart`
- `RentLowQ`
- `RentMedian`
- `RentHighQ`
- `MedRent`
- `MedRentPctHousInc`
- `MedOwnCostPctInc`
- `MedOwnCostPctIncNoMtg`
- `NumInShelters`
- `NumStreet`

### 4.8 流动性与出生地变量

描述人口流动和居住稳定性，例如：

- `PctForeignBorn`
- `PctBornSameState`
- `PctSameHouse85`
- `PctSameCity85`
- `PctSameState85`

### 4.9 警务与执法变量

这组变量来自 `LEMAS` 数据，是该数据集中一个很有特色但缺失也最严重的部分，例如：

- `LemasSwornFT`
- `LemasSwFTPerPop`
- `LemasSwFTFieldOps`
- `LemasSwFTFieldPerPop`
- `LemasTotalReq`
- `LemasTotReqPerPop`
- `PolicReqPerOffic`
- `PolicPerPop`
- `RacialMatchCommPol`
- `PctPolicWhite`
- `PctPolicBlack`
- `PctPolicHisp`
- `PctPolicAsian`
- `PctPolicMinor`
- `OfficAssgnDrugUnits`
- `NumKindsDrugsSeiz`
- `PolicAveOTWorked`
- `PolicCars`
- `PolicOperBudg`
- `LemasPctPolicOnPatr`
- `LemasGangUnitDeploy`
- `LemasPctOfficDrugUn`
- `PolicBudgPerPop`

### 4.10 地理与通勤变量

例如：

- `LandArea`
- `PopDens`
- `PctUsePubTrans`

## 5. 目标变量说明

目标变量为：

- `ViolentCrimesPerPop`

作者说明其含义为：

- 社区每 `10万人` 的暴力犯罪数量
- 原始构成来自 murder、rape、robbery、assault 四类犯罪
- 在最终文件中已经被归一化到 `0-1`

需要特别注意：

- 一些州对 `rape` 的统计口径存在争议
- 因此部分城市在原始暴力犯罪计算中会产生错误
- 这些城市已被作者从最终数据集中剔除
- 被剔除社区中有不少来自美国中西部

这说明数据作者已经做过一定的质量控制，但也意味着样本并不是对所有美国社区的完全覆盖。

## 6. 归一化方式与含义

说明文件中对归一化讲得非常重要，做分析时必须理解清楚。

### 6.1 所有数值变量都被标准化到 `0-1`

作者使用的是一种：

- `Unsupervised`
- `equal-interval binning`

的归一化方式，把所有数值变量压缩到 `0.00-1.00` 区间。

### 6.2 归一化保留了什么

- 同一个变量内部的大致相对大小关系仍然保留
- 例如某社区人口值约为另一个社区的两倍，在归一化后通常仍会体现出这种相对差异

### 6.3 归一化没有保留什么

- 不同变量之间的数值不能直接比较大小
- 例如不能把 `whitePerCap` 的数值和 `blackPerCap` 的数值直接按绝对值比较解释

### 6.4 极端值处理

- 高于均值 `3 SD` 以上的极端值会被压到 `1.00`
- 低于均值 `3 SD` 以下的极端值会被压到 `0.00`

这意味着：

- 变量的极端尾部信息被截断了一部分
- 线性模型中对极端值的解释会受到影响
- 系数可以做方向与相对影响的分析，但不适合直接做原始单位意义下的政策解释

## 7. 缺失值情况

这是该数据集最重要的使用注意事项之一。

### 7.1 数据集存在缺失值

说明文件明确写明：

- `Missing Values? Yes`

### 7.2 缺失主要集中在警务变量

原因包括：

- `LEMAS` 调查主要覆盖至少有 `100` 名警员的警察部门
- 小部门只抽样了一部分
- 因此很多社区没有对应的警务数据

这会导致：

- 一大组警务变量缺失严重
- 如果直接删除含缺失的样本，样本量会大幅下降
- 如果强行保留这些变量，建模会变得复杂

对于课程项目，通常更稳妥的做法是：

- 主模型先使用无缺失或缺失很少的社会经济变量
- 将警务变量作为附加分析或稳健性分析

## 8. 文件中给出的统计信息

`communities.names` 已经直接给出了每个变量的：

- 最小值 `Min`
- 最大值 `Max`
- 均值 `Mean`
- 标准差 `SD`
- 与目标变量的相关系数 `Correl`
- 中位数 `Median`
- 众数 `Mode`
- 缺失个数 `Missing`

这对课程 project 很有帮助，因为你们可以据此：

- 快速筛选和犯罪率关系较强的变量
- 识别潜在多重共线性
- 判断哪些变量缺失过多不适合进入主模型

## 9. 从说明文件能直接看出的高相关变量

根据 summary statistics，和 `ViolentCrimesPerPop` 相关性较强的变量包括：

- 正相关较强：
  - `PctIlleg`
  - `racepctblack`
  - `pctWPubAsst`
  - `FemalePctDiv`
  - `TotalPctDiv`
  - `PctPopUnderPov`
  - `PctUnemployed`
- 负相关较强：
  - `PctKids2Par`
  - `PctFam2Par`
  - `racePctWhite`
  - `PctYoungKids2Par`
  - `PctTeen2Par`
  - `pctWInvInc`
  - `PctPersOwnOccup`

这些变量本身就适合用来做：

- 单变量散点图
- 相关性展示
- 逐步建模或分组建模
- 模型解释与讨论

## 10. 该数据集的优势

### 10.1 很适合教学型回归项目

原因包括：

- 样本量较大，接近 `2000`
- 自变量丰富，便于做模型选择
- 目标变量明确，适合回归任务
- 说明文档详细，便于写报告
- 可以自然展开背景、EDA、建模、诊断、结论等课程要求

### 10.2 解释性较强

变量大多具有明确社会含义，例如：

- 贫困
- 就业
- 教育
- 家庭结构
- 居住条件
- 社区稳定性
- 警务配置

因此不仅能做预测，也很适合做“哪些因素与暴力犯罪更相关”的解释性分析。

## 11. 该数据集的局限性

### 11.1 变量很多，容易多重共线性

很多变量在定义上非常接近，例如：

- `PctFam2Par`、`PctKids2Par`、`PctYoungKids2Par`、`PctTeen2Par`
- `medIncome`、`medFamInc`、`perCapInc`
- 房价、租金、住房拥有率相关变量

因此直接把很多变量一起放进线性模型，系数可能会不稳定。

### 11.2 警务变量缺失非常严重

这会限制“社会变量 + 警务变量”的完整联合建模。

### 11.3 变量均为归一化值

这会影响结果的现实单位解释，特别是：

- 系数不能简单按原始单位解释
- 不同变量之间不能直接按数值大小对比

### 11.4 可能存在伦理与解释风险

部分变量涉及：

- 种族结构
- 收入差异
- 家庭结构

因此在展示和结论部分要谨慎表达：

- 只能说明统计关联
- 不能把相关性直接当作因果关系

## 12. 对课程项目的直接启示

如果将该数据集用于回归分析课程 project，`communities.names` 这份说明文件已经足以支持以下内容：

- 研究背景：社区特征与暴力犯罪率的关系
- 数据介绍：数据来源、样本量、变量分类、目标变量定义
- 探索性分析：根据 summary statistics 和变量含义挑选重点变量画图
- 数据清洗：处理 `?` 缺失值，决定是否舍弃高缺失警务变量
- 模型设定：社会经济变量模型、扩展模型、变量筛选模型
- 诊断分析：残差分析、共线性检验、异常值检验、变换讨论
- 结果解释：重点解释几个有理论意义的变量方向和显著性

## 13. 建议的使用策略

如果你们用这个数据集做课程 project，推荐按下面思路使用：

### 13.1 主分析

使用无缺失或几乎无缺失的变量构建主模型，例如：

- 人口结构
- 收入贫困
- 教育就业
- 家庭结构
- 住房条件

### 13.2 扩展分析

再考虑：

- 加入少量警务变量做扩展模型
- 或单独对有警务数据的子样本建模

### 13.3 诊断重点

重点做：

- 残差分析
- 共线性检验
- 变量变换或响应变量变换
- 模型选择比较

## 14. 一句话总结

`communities.names` 描述的是一个面向“社区暴力犯罪率预测”的经典回归数据集：样本量足够、变量丰富、解释空间强，非常适合做回归分析课程项目；但它也伴随明显的高维、多重共线性和严重缺失问题，因此建模时应重视变量筛选、缺失处理和模型诊断。
