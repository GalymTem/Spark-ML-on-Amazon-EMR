# LAB 6 REPORT  
## Distributed Machine Learning with Spark ML on Amazon EMR  

---

## Overview

This project demonstrates the implementation of a distributed machine learning pipeline using Apache Spark on Amazon EMR.

The objective of this laboratory work is to design and execute a scalable end-to-end Spark ML workflow capable of handling real-world datasets in a distributed cluster environment.

Distributed systems allow computational tasks to be parallelized across multiple nodes, improving performance and scalability. In this experiment, a customer churn prediction model was built and evaluated using Spark ML in a distributed setup.

---

## Lab Goals

- Build a complete Spark ML pipeline
- Perform distributed feature engineering
- Train and evaluate a classification model
- Submit Spark jobs to Amazon EMR
- Observe distributed execution behavior
- Analyze the impact of feature engineering

---

## Amazon EMR Cluster Configuration

The Spark job was executed on an Amazon EMR cluster with the following configuration:

- **Deployment Mode:** Instance Groups
- **Primary Node:** 1 × m4.large
- **Core Nodes:** 2 × m4.large
- **Applications Installed:**
  - Hadoop
  - Spark
- **IAM Roles:** Default EMR roles
- 
<img width="900" height="359" alt="image" src="https://github.com/user-attachments/assets/cfbbc4c2-7a13-4fdf-b986-f1adc8102c60" />

The cluster used YARN for distributed resource management.

---

## Dataset Information

The **Bank Customer Churn Dataset** was used for this experiment.

<img width="900" height="421" alt="image" src="https://github.com/user-attachments/assets/77e2daeb-33f8-4511-bc3a-21ad6c5792ae" />


The dataset includes customer banking attributes used to predict whether a customer will leave the bank.



### Target Variable

- `Exited`
  - 0 → Customer stays  
  - 1 → Customer churns  

The dataset was uploaded to HDFS to enable distributed access across cluster nodes.

---

## Uploading Data to HDFS

The following steps were performed:
<img width="776" height="129" alt="image" src="https://github.com/user-attachments/assets/0485f685-6e95-47c8-906a-338d73d30d85" />

<img width="704" height="89" alt="image" src="https://github.com/user-attachments/assets/a5b125e4-d924-4fd4-85e3-b51bf13fb431" />

### Result:
<img width="167" height="36" alt="image" src="https://github.com/user-attachments/assets/ad477c8a-0e2b-4a2c-8ad8-a69a1fd8d7d0" />

