# dl-rerank (alpha)
Deep learning powered personalized re-ranking solution

#### User Interest Modeling Strategy
Given item list needed for ranking, we use DIN (deep interest network) modeling user diverse interest, DIEN is another good solution for this problem, the problem with this solution is we need to do lots of engineering optimization to get good performance when we use RNN, may be SRU is a candidate solution.

**Reference**
> [DIN](https://arxiv.org/abs/1706.06978): Deep Interest Network for Click-Through Rate Prediction <br />
> [DIEN](https://arxiv.org/abs/1809.03672): Deep Interest Evolution Network for Click-Through Rate Prediction <br />
> [SRU](https://arxiv.org/abs/1709.02755): Simple Recurrent Units for Highly Parallelizable Recurrence <br />

#### Item Modeling Strategy
After modeling user interest, given item targeted user vectorized representation and item list vectorized representation, and item click or not click label info. To precisely model (personalized user representation, item representation, context, label) relation, we need to consider item list info, with this info, we can compute each (personalized user representation, item representation)'s precise vectorized representation. Considering the computation budget we can apply dense tranformation before apply Transformer to do self-attention. We could use transformer to do user interest modeling also (BST).

**Reference**
>[Transformer](https://arxiv.org/abs/1706.03762): Attention Is All You Need <br />
>[PRM-Rerank](https://arxiv.org/abs/1904.06813): Personalized Re-ranking for Recommendation <br />
>[BST](https://arxiv.org/abs/1905.06874): Behavior Sequence Transformer for E-commerce Recommendation in Alibaba <br />

#### Engineering Related
> **Embedding**: support share embedding <br />
> **MBA**: support mini-batch aware regularization for sparse categorical feature <br />
> **XLA**: support xla <br />
> **Mixed Precision**: support mixed precision, **this feature can only be used with tf >=2.2.0** <br />
> **Distributed Training**: support parameter-server distributed training strategy <br />
