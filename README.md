# MemN2N-tensorflow
TensorFlow implementation of an [End-to-End Memory Network [1]](http://arxiv.org/abs/1503.08895) for the bAbI dataset [2].

### Requirements
* Python3
* Tensorflow r0.11

### Dataset
Download the [bAbI Tasks Data 1-20 (v1.2)](http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz). Please adjust the path to the dataset in the main runfile. 
For further description of the dataset see the [bAbI project website](https://research.fb.com/downloads/babi/).

### References

[1] S. Sukhbaatar, J. Weston, R. Fergus et al., “End-to-end memory networks,” in Advances in neural
information processing systems, 2015, pp. 2440–2448.

[2] J. Weston, A. Bordes, S. Chopra, A. M. Rush, B. van Merriënboer, A. Joulin, and T. Mikolov,
“Towards ai-complete question answering: A set of prerequisite toy tasks,” arXiv preprint
arXiv:1502.05698, 2015.