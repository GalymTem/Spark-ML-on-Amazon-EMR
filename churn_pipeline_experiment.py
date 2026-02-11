from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def build_spark(app_name: str) -> SparkSession:
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_and_clean(spark: SparkSession, hdfs_path: str):
    df = spark.read.csv(hdfs_path, header=True, inferSchema=True)

    # Drop typical ID/text columns if they exist (dataset variants differ)
    for c in ["RowNumber", "CustomerId", "Surname"]:
        if c in df.columns:
            df = df.drop(c)

    # Ensure label is numeric
    if "Exited" in df.columns:
        df = df.withColumn("Exited", col("Exited").cast("double"))

    return df


def evaluator(metric: str):
    return MulticlassClassificationEvaluator(
        labelCol="Exited",
        predictionCol="prediction",
        metricName=metric
    )


def evaluate(df_pred):
    metrics = ["accuracy", "weightedPrecision", "weightedRecall", "f1"]
    out = {m: evaluator(m).evaluate(df_pred) for m in metrics}

    # Simple confusion matrix (Pred x Label counts)
    cm = (
        df_pred
        .groupBy("Exited", "prediction")
        .count()
        .orderBy(col("Exited"), col("prediction"))
    )
    return out, cm


def make_baseline_pipeline():
    geo_index = StringIndexer(
        inputCol="Geography", outputCol="GeoIdx", handleInvalid="keep"
    )
    gen_index = StringIndexer(
        inputCol="Gender", outputCol="GenIdx", handleInvalid="keep"
    )

    ohe = OneHotEncoder(
        inputCols=["GeoIdx", "GenIdx"],
        outputCols=["GeoOHE", "GenOHE"]
    )

    numeric = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    assembler = VectorAssembler(
        inputCols=numeric + ["GeoOHE", "GenOHE"],
        outputCol="features"
    )

    scaler = StandardScaler(
        inputCol="features", outputCol="featuresScaled", withMean=True, withStd=True
    )

    clf = LogisticRegression(
        labelCol="Exited", featuresCol="featuresScaled", maxIter=50, regParam=0.0
    )

    return Pipeline(stages=[geo_index, gen_index, ohe, assembler, scaler, clf])


def make_ablation_pipeline():
    numeric = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]

    assembler = VectorAssembler(
        inputCols=numeric,
        outputCol="features"
    )

    scaler = StandardScaler(
        inputCol="features", outputCol="featuresScaled", withMean=True, withStd=True
    )

    clf = LogisticRegression(
        labelCol="Exited", featuresCol="featuresScaled", maxIter=50, regParam=0.0
    )

    return Pipeline(stages=[assembler, scaler, clf])


def main():
    spark = build_spark("CustomerChurn_Baseline_Ablation")

    path = "hdfs:///user/hadoop/churn_input/Churn_Modelling.csv"
    print("\n=== Loading dataset ===")
    data = load_and_clean(spark, path)

    # Fair comparison: SAME split for both models
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    print(f"Train rows: {train.count()} | Test rows: {test.count()}")

    print("\n=== Baseline (with categorical features) ===")
    baseline_model = make_baseline_pipeline().fit(train)
    baseline_pred = baseline_model.transform(test)
    baseline_metrics, baseline_cm = evaluate(baseline_pred)

    print("Baseline metrics:", baseline_metrics)
    print("Baseline confusion matrix:")
    baseline_cm.show(truncate=False)

    print("\n=== Ablation (numeric only) ===")
    ablation_model = make_ablation_pipeline().fit(train)
    ablation_pred = ablation_model.transform(test)
    ablation_metrics, ablation_cm = evaluate(ablation_pred)

    print("Ablation metrics:", ablation_metrics)
    print("Ablation confusion matrix:")
    ablation_cm.show(truncate=False)

    print("\n=== Comparison summary ===")
    diff_acc = baseline_metrics["accuracy"] - ablation_metrics["accuracy"]
    print(f"Accuracy baseline: {baseline_metrics['accuracy']:.4f}")
    print(f"Accuracy ablation : {ablation_metrics['accuracy']:.4f}")
    print(f"Accuracy drop     : {diff_acc:.4f}")

    spark.stop()


if __name__ == "__main__":
    main()
