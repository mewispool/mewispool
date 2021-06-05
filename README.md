# Maximum Entroy Weighted Independent Set Pooling for Graph Neural Networks (MEWISPool)

![image](img/mewispool.jpg)

In this paper, we propose a novel pooling layer for graph neural networks based
on maximizing the mutual information between the pooled graph and the input
graph. Since the maximum mutual information is difficult to compute, we em-
ploy the Shannon capacity of a graph as an inductive bias to our pooling method.
More precisely, we show that the input graph to the pooling layer can be viewed
as a representation of a noisy communication channel. For such a channel, send-
ing the symbols belonging to an independent set of the graph yields a reliable
and error-free transmission of information. We show that reaching the maximum
mutual information is equivalent to finding a maximum weight independent set of
the graph where the weights convey entropy contents. Through this communica-
tion theoretic standpoint, we provide a distinct perspective for posing the problem
of graph pooling as maximizing the information transmission rate across a noisy
communication channel, implemented by a graph neural network. We evaluate
our method, referred to as Maximum Entropy Weighted Independent Set Pooling
(MEWISPool), on graph classification tasks and the combinatorial optimization
problem of the maximum independent set. Empirical results demonstrate that our
method achieves the state-of-the-art and competitive results on graph classification
tasks and the maximum independent set problem in several benchmark datasets.