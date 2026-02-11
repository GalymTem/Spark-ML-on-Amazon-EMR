# CustomerChurn_Baseline_Ablation
# LAB 6 REPORT  
## Distributed Machine Learning with Spark ML on Amazon EMR  

---

## 📖 Overview

This project demonstrates the implementation of a distributed machine learning pipeline using Apache Spark on Amazon EMR.

The objective of this laboratory work is to design and execute a scalable end-to-end Spark ML workflow capable of handling real-world datasets in a distributed cluster environment.

Distributed systems allow computational tasks to be parallelized across multiple nodes, improving performance and scalability. In this experiment, a customer churn prediction model was built and evaluated using Spark ML in a distributed setup.

---

## 🎯 Lab Goals

- Build a complete Spark ML pipeline
- Perform distributed feature engineering
- Train and evaluate a classification model
- Submit Spark jobs to Amazon EMR
- Observe distributed execution behavior
- Analyze the impact of feature engineering

---

## ☁️ Amazon EMR Cluster Configuration

The Spark job was executed on an Amazon EMR cluster with the following configuration:

- **Deployment Mode:** Instance Groups
- **Primary Node:** 1 × m4.large
- **Core Nodes:** 2 × m4.large
- **Applications Installed:**
  - Hadoop
  - Spark
- **IAM Roles:** Default EMR roles

The cluster used YARN for distributed resource management.

---

## 📊 Dataset Information

The **Bank Customer Churn Dataset** was used for this experiment.

The dataset includes customer banking attributes used to predict whether a customer will leave the bank.

### Features Used

- CreditScore  
- Geography  
- Gender  
- Age  
- Tenure  
- Balance  
- NumOfProducts  
- EstimatedSalary  

### Target Variable

- `Exited`
  - 0 → Customer stays  
  - 1 → Customer churns  

The dataset was uploaded to HDFS to enable distributed access across cluster nodes.

---

## 📂 Uploading Data to HDFS

The following steps were performed:

```bash
hdfs dfs -mkdir -p /user/hadoop/churn_input
hdfs dfs -put Churn_Modelling.csv /user/hadoop/churn_input/
hdfs dfs -ls /user/hadoop/churn_input
