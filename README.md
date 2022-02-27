# Fair-SSL
Source Code for ICSE 2022 Paper - **Fair-SSL: Building fair ML Software with less data**

Ethical bias in machine learning models has become a matter of concern in the software engineering community. Most of the prior software engineering works concentrated on finding ethical bias in models rather than fixing it. After finding bias, the next step is mitigation. Prior researchers mainly tried to use supervised approaches to achieve fairness. However, in the real world, getting data with trustworthy ground truth is challenging and also ground truth can contain human bias. Semi-supervised learning is a domain of machine learning where labeled and unlabeled data both are used to overcome the data labeling challenges. We, in this work, applied four popular semi-supervised techniques as pseudo-labelers to create fair classification models. Our framework, Fair-SSL,  takes a very small amount (10\%) of labeled data as input and generates pseudo-labels for the unlabeled data. We then synthetically generate new data points to balance the training data based on class and protected attribute as proposed by Chakraborty et al. in FSE 2021. Finally, classification model is trained on the balanced pseudo-labeled data and validated on test data. After experimenting on ten datasets and three learners, we found out that Fair-SSL achieves similar performance like three other state-of-the-art bias mitigation algorithms. Where prior algorithms require much training data, Fair-SSL requires only 10\% of the labeled training data. As per our knowledge, this is the first SE work where semi-supervised techniques are used to fight against ethical bias in ML models.


## Dataset Description - 

1> Adult Income dataset - http://archive.ics.uci.edu/ml/datasets/Adult

2> COMPAS - https://github.com/propublica/compas-analysis

3> German Credit - https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29 

4> Default Credit - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

5> Heart - https://archive.ics.uci.edu/ml/datasets/Heart+Disease

6> Bank Marketing - https://archive.ics.uci.edu/ml/datasets/bank+marketing

7> Home Credit - https://www.kaggle.com/c/home-credit-default-risk

8> Student Performance - https://archive.ics.uci.edu/ml/datasets/Student+Performance

9> MEPS15 - https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181

10> MEPS16 - https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-192



## Data Preprocessing -
* We have used data preprocessing as suggested by [IBM AIF360](https://github.com/IBM/AIF360)
* The rows containing missing values are ignored, continuous features are converted to categorical (e.g., age<25: young,age>=25: old), non-numerical features are converted to numerical(e.g., male: 1, female: 0). Fiinally, all the feature values are normalized(converted between 0 to 1). 
* For `optimized Pre-processing`, plaese visit [Optimized Preprocessing](https://github.com/Trusted-AI/AIF360/blob/master/aif360/algorithms/preprocessing/optim_preproc.py)
* For `Fair-SMOTE`, please visit [Fair-SMOTE](https://github.com/joymallyac/Fair-SMOTE)
* For `Fairway`, please visit [Fairway](https://github.com/joymallyac/Fairway)

## Tutorial to run the code

```git clone https://github.com/senthusiast/Fair-SSL```

```cd Fair-SSL```

```pip install requirements.txt```

## Download Datasets

We could not upload MEPS15 and MEPS16 because of space restrictions. Please download them [MEPS15](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181) , [MEPS16](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-192)

## For RQ1, \& RQ2

*Self-Training*

```cd Self-Training```

```python Self-Training.py```

*LabelSpreading*

```cd LabelSpreading```

```python LabelSpreading.py```

*LabelPropagation*

```cd LabelPropagation```

```python LabelPropagation.py```

*Co-Training*

```cd Co-Training```

```python Co-Training.py```

## For RQ3

```cd Initial-Set-Variation```

```python Initial_Set_Variation.py```
