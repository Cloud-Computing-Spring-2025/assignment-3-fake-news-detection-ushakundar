from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, concat_ws, regexp_replace, rand, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# === 1. Start Spark Session ===
spark = SparkSession.builder.appName("FakeNewsClassifier").getOrCreate()

# === 2. Load and Clean Dataset ===
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)
df = df.na.drop(subset=["title", "text", "label"])
df.createOrReplaceTempView("news_data")

# === 3. Combine Title + Text and Clean Label Leakage ===
df_cleaned = df.withColumn("text", lower(concat_ws(" ", "title", "text")))
df_cleaned = df_cleaned.withColumn("text", regexp_replace("text", r"\bfake\b|\breal\b", ""))

# === 4. Add Label Noise (~5% of rows) ===
df_cleaned = df_cleaned.withColumn(
    "label",
    when(rand() < 0.05,
         when(df_cleaned["label"] == "fake", "real").otherwise("fake")
    ).otherwise(df_cleaned["label"])
)

# Save preview
df_cleaned.limit(5).toPandas().to_csv("task1_output.csv", index=False)

# === 5. Train-Test Split ===
train_df, test_df = df_cleaned.randomSplit([0.8, 0.2], seed=42)

# === 6. Feature Engineering ===
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=1000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
indexer = StringIndexer(inputCol="label", outputCol="label_index")

# Train data processing
train_tok = tokenizer.transform(train_df)
train_clean = remover.transform(train_tok)
train_feat = hashingTF.transform(train_clean)
idf_model = idf.fit(train_feat)
train_scaled = idf_model.transform(train_feat)
label_indexer_model = indexer.fit(train_scaled)
train_final = label_indexer_model.transform(train_scaled)

# Test data processing (same models)
test_tok = tokenizer.transform(test_df)
test_clean = remover.transform(test_tok)
test_feat = hashingTF.transform(test_clean)
test_scaled = idf_model.transform(test_feat)
test_final = label_indexer_model.transform(test_scaled)

# Save feature preview
train_final.select("id", "filtered_words", "features", "label_index") \
    .limit(5).toPandas().to_csv("task3_output.csv", index=False)

# === 7. Train Model ===
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
model = lr.fit(train_final)

# === 8. Predict ===
predictions = model.transform(test_final)
predictions.select("id", "title", "label_index", "prediction") \
    .toPandas().to_csv("task4_output.csv", index=False)

# === 9. Evaluate ===
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")

accuracy = evaluator_acc.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score"],
    "Value": [accuracy, f1_score]
}).to_csv("task5_output.csv", index=False)

# === 10. Done ===
print("All tasks completed.")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")
