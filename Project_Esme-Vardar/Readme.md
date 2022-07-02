# Mutual CRF-GNN for Few-shot Learning

This readme file is an outcome of the [CENG502 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2022) Project List]([https://github.com/sinankalkan/CENG502-Spring2021](https://github.com/CENG502-Projects/CENG502-Spring2022)) for a complete list of all paper reproduction projects.

# 1. Introduction

We chose Mutual CRF-GNN for Few-shot Learning  to implement as a term project within the scope of the CENG502 Advanced Deep Learning course. This paper is CVPR 2021 paper. We aim to achieve the same results as researchers by implementing this paper, which does not have open source code. Thus, we hope to create an open source for those who want to apply the method in the paper.

## 1.1. Paper summary

This paper brings a new method to GNN. Affinity is a critical part of GNN, and in general affinity is computed in feature space, and 
does not take fully advantage of semantic labels associated to these features.In this paper CRF(Conditional Random Field) is used on labels and features of support data to infer a affinity in the label space. This method is called as Mutual CRF-GNN (MCGN). Their results show that this method outperforms state of art methods on 5-way 1-shot and 5-way5-shot setting on datasets miniImageNet, tieredImageNet, and
CIFAR-FS.

# 2. The method and my interpretation

## 2.1. The original method


@TODO: Explain the original method.

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
