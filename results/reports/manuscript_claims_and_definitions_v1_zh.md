# 论文任务、标签与结果解释规范 v1

## 研究任务

本文的任务是根据PCAOB已经公开并确认属于inspection deficiency的finding narrative，预测与其PCAOB Audit Area字段相对应的研究用粗粒度类别，并将不适合自动分类的案例转交进一步复核。模型不负责发现审计缺失，不判断财务报表是否错报，不判断企业是否舞弊，也不决定审计意见是否恰当。

推荐英文定义为：

> This study predicts coarse-grained research categories derived from PCAOB-assigned audit areas using inspection-finding narratives. It does not independently discover audit deficiencies, determine an objectively unique audit-area classification, or replace professional audit judgment.

## 标签层级

标签生成过程包含三个不同层级。`Audit Area`是PCAOB数据中的原始字段；`target_level2`主要对原始名称进行确定性的标准化和别名合并；`target_level1`由本文预先制定的映射规则把216个标准化细类映射到12个粗粒度研究类别。`target_level1`不是PCAOB官方taxonomy，也不是由审计专家共识建立的标准。

## 正确性定义

本文的prediction correctness表示模型预测与PCAOB-derived `target_level1`一致，即$\hat y_i=y_i^{PCAOB-derived}$。它不表示模型预测与某个独立、客观且唯一的专业真值一致。因此正文宜使用`agreement with the PCAOB-derived label`、`label disagreement`和`classification error under the study taxonomy`，避免使用`true audit category`或把全部不一致解释为模型不理解审计。

推荐英文限定语为：

> Prediction correctness denotes agreement with the PCAOB-derived research label and should not be interpreted as independent professional validation of the underlying audit area.

## Risk router的目标

Risk router估计的是SEC-BERT预测与PCAOB-derived标签不一致的风险，而不是判断官方标签是否客观正确。该风险可能来自模型失误、低margin、多主题竞争、上下文缺失、taxonomy边界或标签不一致。被转交样本应称为`high-risk or potentially ambiguous cases`，而不是直接称为`model errors`。

推荐英文定义为：

> The router estimates the risk that a SEC-BERT prediction will disagree with the PCAOB-derived study label. The estimated risk may reflect model uncertainty, incomplete textual evidence, competing audit areas, taxonomy boundaries, or label inconsistency; it does not adjudicate whether the official label is objectively correct.

## 人工转交的含义

转交是系统的合理输出，而不是分类失败的同义词。当前实验测量转交比例、自动接受区的一致率和被转交队列中包含多少label disagreements。研究没有实测审计师是否能修正全部转交案例、人工耗时或最终人机系统准确率，因此不得把所有转交案例假设为经人工后100%正确。

## LLM与RAG的角色

Direct LLM和Report-RAG不是标签裁判。它们与SEC-BERT一样受到输入文本、上下文可见性和研究taxonomy的限制。LLM/RAG与PCAOB-derived标签不一致时，应区分明显分类错误、证据真实但推理错误、audit area与deficiency mechanism混淆、多类别均合理、不可见上下文以及taxonomy映射差异。现阶段它们只属于困难案例辅助、证据支持和harmful-takeover分析，不属于已验证的自动接管路径。

## 标签歧义的客观证据

原始数据包含615个多audit-area findings；268组finding具有完全相同的标准化文本，其中169组对应不同标签集合；245个确定性目标冲突样本已从单标签任务中排除。另有1,295笔文本包含可能依赖前文的表达，其中366笔补回同一report、issuer和audit-area的共享前言，929笔仍缺少可用前言。这些现象表明观察到的错误率综合了模型失误、输入不可识别性、taxonomy边界和潜在标签不一致。

## 新增分组分析后的解释

上下文不足与错误风险的关系不是单调且跨分割稳定的。Clean development OOF中，`context_unresolved`错误率为8.22%，约为不存在该标记样本的1.29倍；`context_added`错误率仅3.60%。但在retrospective test中，`context_unresolved`组accuracy为96.21%，高于`original_sufficient`组的92.41%。这可能受到样本量和类别构成影响，因此论文只能把上下文不足写成潜在风险来源，不能声称已经证明它必然降低准确率。

Development OOF中更强的描述性信号是模型分歧：SEC-BERT与TF-IDF不一致的478笔样本错误率为33.47%，约为一致组的9.55倍；`multi_topic`样本错误率为9.13%，约为其余样本的1.54倍。它们支持把模型分歧和多主题竞争作为referral特征，但仍不构成标签歧义的专业判定。

在120笔分层PCAOB RAG pilot中，Report-RAG相对SEC-BERT总体为11 beneficial、22 harmful。具有`context_needs_context`标记的29笔中为1 beneficial、8 harmful；`context_unresolved`的26笔中为1 beneficial、5 harmful。由于样本是刻意分层抽取的，这些数字只能描述失效机制，不能估计母体发生率。

## 结果表述模板

推荐写法：

> After referring 8.54% of high-risk or potentially ambiguous cases, the remaining 91.46% of automatically accepted predictions achieved 97.67% agreement with the PCAOB-derived coarse-grained labels.

不推荐写法：

> The system identified the true categories of audit errors with 97.67% accuracy.

推荐写法：

> The observed disagreement rate combines model failures, incomplete textual evidence, taxonomy-boundary ambiguity, and possible label inconsistency.

不推荐写法：

> SEC-BERT failed to understand 6.63% of the audit findings.

## 研究者主导错误分析

如果没有双审计专家，可以进行`single-author exploratory adjudication`，判断文本是否信息充分、是否存在多个合理类别、是否依赖完整报告以及不一致属于何种机制。该分析必须保持探索性，不得把研究者个人判断改成新的ground truth，也不得据此修改PCAOB-derived标签后重新计算主结果。双专家独立复核和一致性统计应列为未来验证。

