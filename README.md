# Text Classification using Support Vector Machines (SVM)

A comprehensive text classification system using SVM with TF-IDF vectorization. Works with external CSV datasets and provides complete performance metrics.

## 📁 Project Files

- **`svm_text_classifier.py`** - Main program (single file solution)
- **`sample_dataset.csv`** - Sample dataset for testing
- **`requirements.txt`** - Python dependencies
- **`README.md`** - This file

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with Sample Data

```bash
python svm_text_classifier.py sample_dataset.csv
```

### 3. Run with Your Own Data

```bash
python svm_text_classifier.py your_data.csv
```

## 📊 CSV Format Required

Your CSV file must have two columns: `text` and `label`

**Example:**
```csv
text,label
"The new smartphone has great features",Technology
"The team won the championship",Sports
"Eating healthy improves wellness",Health
"The restaurant serves delicious food",Food
```

## 🎯 What You Get

### Core Metrics
✅ **Accuracy** - Overall correctness percentage
✅ **Precision** - How many predicted positives are correct
✅ **Recall** - How many actual positives were found
✅ **F1-Score** - Harmonic mean of precision and recall

### Advanced Analysis
✅ **Confusion Matrix** - Visual representation with analysis
✅ **Per-Category Metrics** - Detailed breakdown for each category
✅ **Error Analysis** - Identifies most confused category pairs
✅ **Cross-Validation** - 5-fold CV for robust evaluation
✅ **Sample Predictions** - Shows actual vs predicted examples

### Dataset Insights
✅ **Category Distribution** - Visual bar charts
✅ **Balance Ratio** - Checks dataset balance
✅ **Text Statistics** - Word count analysis
✅ **Split Distribution** - Train/test breakdown

## 📈 Example Output

```
=====================================================================================
  TEXT CLASSIFICATION USING SVM - ENHANCED VERSION
=====================================================================================

📊 STEP 2: Dataset Analysis
-------------------------------------------------------------------------------------
✓ Number of categories: 4
✓ Categories: Food, Health, Sports, Technology

📈 Category Distribution:
  Food           :   15 samples ( 25.0%) ████████████
  Health         :   15 samples ( 25.0%) ████████████
  Sports         :   15 samples ( 25.0%) ████████████
  Technology     :   15 samples ( 25.0%) ████████████

✓ Dataset balance ratio: 1.00 (1.0 = perfectly balanced)

🤖 STEP 5: Training SVM Model
-------------------------------------------------------------------------------------
✓ Model training completed!

🔄 Performing 5-Fold Cross-Validation...
✓ CV Accuracy: 0.5111 (+/- 0.3014)

📈 STEP 6: Model Evaluation & Metrics
-------------------------------------------------------------------------------------

🎯 OVERALL PERFORMANCE METRICS
  Metric                    Score      Percentage    Quality
  --------------------------------------------------------------------------------
  Accuracy                  0.5333     53.33%        Needs Improvement ⭐
  Precision (weighted)      0.6190     61.90%        Fair ⭐⭐
  Recall (weighted)         0.5333     53.33%        Needs Improvement ⭐
  F1-Score (weighted)       0.5263     52.63%        Needs Improvement ⭐

📋 Detailed Classification Report
              precision    recall  f1-score   support
        Food       0.67      0.50      0.57         4
      Health       0.43      1.00      0.60         3
      Sports       1.00      0.50      0.67         4
  Technology       0.33      0.25      0.29         4

🔢 Confusion Matrix
           Predicted
              Food   Health   Sports   Technology
Actual   Food    2        0        0        2
Actual Health    0        3        0        0
Actual Sports    0        2        2        0
Actual Tech      1        2        0        1

🔍 Error Analysis
✓ Total errors: 7 out of 15 predictions
✓ Error rate: 46.67%

📉 Most Confused Category Pairs:
  Food → Technology: 2 times
  Sports → Health: 2 times
  Technology → Health: 2 times
```

## 📊 Output Files

After running, you'll get:
- **Console output** - Complete analysis displayed
- **`results_detailed.txt`** - Summary metrics saved to file

## 🔧 Requirements

- Python 3.7+
- pandas
- scikit-learn
- numpy

Install with:
```bash
pip install -r requirements.txt
```

## 💡 Features

### Quality Ratings
- ⭐⭐⭐⭐⭐ Excellent (90%+)
- ⭐⭐⭐⭐ Very Good (80-90%)
- ⭐⭐⭐ Good (70-80%)
- ⭐⭐ Fair (60-70%)
- ⭐ Needs Improvement (<60%)

### Multiple Averages
- **Macro**: Unweighted mean (treats all categories equally)
- **Micro**: Total true positives (good for imbalanced data)
- **Weighted**: Weighted by support (accounts for class imbalance)

### Cross-Validation
5-fold cross-validation shows model stability across different data splits. Lower standard deviation indicates more stable model.

## 🎓 Perfect For

- ✅ Academic assignments
- ✅ Research projects
- ✅ Text classification tasks
- ✅ Performance analysis
- ✅ Model evaluation

## 📝 How It Works

1. **Load Dataset** - Reads CSV and validates format
2. **Analyze Data** - Shows distribution and statistics
3. **Split Data** - 75% training, 25% testing
4. **Vectorize** - Converts text to TF-IDF features (500 features, bigrams)
5. **Train** - Trains Linear SVM classifier
6. **Cross-Validate** - 5-fold CV for robust evaluation
7. **Evaluate** - Calculates all metrics
8. **Analyze Errors** - Identifies confusion patterns
9. **Report** - Displays comprehensive results
10. **Save** - Saves summary to file

## 🔍 Understanding Metrics

### Accuracy
Percentage of correct predictions out of total predictions.
```
Accuracy = (Correct Predictions) / (Total Predictions)
```

### Precision
Of all items predicted as positive, how many are actually positive?
```
Precision = True Positives / (True Positives + False Positives)
```

### Recall
Of all actual positive items, how many did we find?
```
Recall = True Positives / (True Positives + False Negatives)
```

### F1-Score
Harmonic mean of precision and recall (balanced metric).
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

## 🛠️ Customization

You can modify these parameters in the code:

```python
# Test size (line ~120)
test_size=0.25  # Change to 0.2 for 80-20 split

# TF-IDF features (line ~150)
max_features=500,  # Increase for more features
ngram_range=(1, 2),  # (1,1) for unigrams only

# SVM parameters (line ~180)
C=1.0,  # Regularization strength
max_iter=3000,  # Maximum iterations

# Cross-validation (line ~200)
cv=5  # Number of folds
```

## 📊 Sample Dataset

The included `sample_dataset.csv` contains:
- 60 documents
- 4 categories (Technology, Sports, Health, Food)
- 15 samples per category
- Balanced distribution

Perfect for testing and learning!

## 🚨 Troubleshooting

### Error: "File not found"
- Check file path is correct
- Ensure CSV file is in same directory

### Error: "CSV must have 'text' and 'label' columns"
- Verify column names are exactly: `text` and `label`
- Check CSV format is correct

### Low Accuracy
- Add more training samples (50+ per category recommended)
- Ensure categories are distinct
- Check data quality
- Try adjusting parameters

## 📈 Tips for Better Results

1. **Balanced Dataset** - Similar number of samples per category
2. **Quality Text** - Clean, relevant text improves results
3. **Sufficient Data** - Aim for 50+ samples per category
4. **Distinct Categories** - Categories should have clear differences
5. **More Features** - Increase `max_features` for complex tasks

## 🎯 Expected Performance

Performance depends on:
- Dataset size (more data = better accuracy)
- Category distinctiveness (clear differences = better results)
- Text quality (clean, relevant text = better performance)

**Typical Results:**
- Small dataset (50-100 samples): 50-70% accuracy
- Medium dataset (500-1000 samples): 70-85% accuracy
- Large dataset (5000+ samples): 85-95% accuracy

## 📞 Support

If you encounter issues:
1. Verify CSV format matches requirements
2. Check all required packages are installed
3. Ensure dataset has at least 10 samples
4. Try with provided `sample_dataset.csv` first

## ✅ Summary

**Single file. One command. Complete analysis.**

```bash
python svm_text_classifier.py your_data.csv
```

Get comprehensive text classification with:
- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ Confusion Matrix with Analysis
- ✅ Per-Category Detailed Metrics
- ✅ Error Analysis
- ✅ Cross-Validation
- ✅ Sample Predictions
- ✅ Dataset Insights

**Perfect for academic submissions and text classification projects!**

---

**Version**: 1.0
**Author**: Text Classification using SVM
**License**: Educational Use
