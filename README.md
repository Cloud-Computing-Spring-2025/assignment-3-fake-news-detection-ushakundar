# Assignment-5-FakeNews-Detection

## Project Overview
This project builds a binary text classification pipeline using Apache Spark MLlib to detect whether a news article is FAKE or REAL based on its content. The pipeline processes and vectorizes text data, applies machine learning for classification, and evaluates the model’s performance.

## Dataset
File: fake_news_sample.csv

### Columns:

1) id: Article ID

2) title: Title of the article

3) text: Main content

4) label: Ground truth (FAKE or REAL)

## Environment Setup
Ensure the following Python libraries are installed:
```bash
pip install pyspark pandas
```
Run using:
```bash
spark-submit main.py
```

## Task Breakdown

### Task 1: Load & Basic Exploration

1. Load fake_news_sample.csv into Spark DataFrame (with schema inferred).

2. Create temporary view news_data.

#### Perform:

1. SHOW first 5 rows

2. Count total articles

3. Display distinct labels

Output: task1_output.csv

### Task 2: Text Preprocessing
1. Convert text to lowercase.

2. Tokenize using Tokenizer.

3. Remove stopwords using StopWordsRemover.

4. Register DataFrame as cleaned_news.

Output: task2_output.csv

### Task 3: Feature Extraction
1. Apply HashingTF and IDF to tokenized text.

2. Use StringIndexer to convert label (FAKE/REAL) into numeric.

3. Optionally use VectorAssembler to assemble feature columns.

Output: task3_output.csv

### Task 4: Model Training
1. Split data into 80% train / 20% test.

2. Train LogisticRegression model.

3. Generate predictions on test data.

Output: task4_output.csv

### Task 5: Evaluation
1. Use MulticlassClassificationEvaluator to calculate:
Accuracy
F1 Score

Output: task5_output.csv

## Output Files
Each task will output a CSV in the root directory:

task1_output.csv

task2_output.csv

task3_output.csv

task4_output.csv

task5_output.csv

## How to Run
Place fake_news_sample.csv in the input/ directory.
Run the full pipeline:
```bash
spark-submit main.py
```
Outputs will be saved as CSV files per task.
