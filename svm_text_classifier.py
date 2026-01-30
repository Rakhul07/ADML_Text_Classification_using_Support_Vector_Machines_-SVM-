"""
Enhanced Text Classification using Support Vector Machines (SVM)
Single file - Works with external CSV dataset
Provides comprehensive metrics and analysis

Usage:
    python svm_classifier_enhanced.py your_dataset.csv

CSV Format Required:
    text,label
    "your text here","category1"
    "another text","category2"
"""

import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 85)
    print(f"  {title}")
    print("=" * 85)


def print_section(title):
    """Print formatted section"""
    print(f"\n{title}")
    print("-" * 85)


def load_dataset(filepath):
    """Load dataset from CSV file"""
    print_section("📂 STEP 1: Loading Dataset")
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ File loaded successfully: {filepath}")
        
        if 'text' not in df.columns or 'label' not in df.columns:
            print("\n❌ ERROR: CSV must have 'text' and 'label' columns")
            print("Expected format:")
            print("  text,label")
            print('  "your text here","category1"')
            sys.exit(1)
        
        df = df.dropna()
        print(f"✓ Total samples: {len(df)}")
        print(f"✓ Columns found: {list(df.columns)}")
        
        return df
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        sys.exit(1)


def analyze_dataset(df):
    """Comprehensive dataset analysis"""
    print_section("📊 STEP 2: Dataset Analysis")
    
    texts = df['text'].values
    labels = df['label'].values
    categories = sorted(df['label'].unique())
    
    print(f"✓ Number of categories: {len(categories)}")
    print(f"✓ Categories: {', '.join(map(str, categories))}")
    print()
    
    # Category distribution
    print("📈 Category Distribution:")
    label_counts = Counter(labels)
    for cat in categories:
        count = label_counts[cat]
        percentage = (count / len(labels)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {cat:15s}: {count:4d} samples ({percentage:5.1f}%) {bar}")
    
    # Check balance
    counts = list(label_counts.values())
    balance_ratio = min(counts) / max(counts) if max(counts) > 0 else 0
    print(f"\n✓ Dataset balance ratio: {balance_ratio:.2f} (1.0 = perfectly balanced)")
    if balance_ratio < 0.5:
        print("  ⚠️  Warning: Imbalanced dataset may affect performance")
    
    # Text statistics
    print(f"\n📝 Text Statistics:")
    text_lengths = [len(str(text).split()) for text in texts]
    print(f"  Average words per text: {np.mean(text_lengths):.1f}")
    print(f"  Min words: {np.min(text_lengths)}")
    print(f"  Max words: {np.max(text_lengths)}")
    print(f"  Median words: {np.median(text_lengths):.1f}")
    
    # Sample texts
    print(f"\n📄 Sample Texts (one per category):")
    for cat in categories[:5]:
        sample = df[df['label'] == cat]['text'].iloc[0]
        print(f"  [{cat}] {sample[:65]}...")
    
    return texts, labels, categories


def prepare_data(texts, labels, test_size=0.25):
    """Split data with stratification"""
    print_section("🔀 STEP 3: Splitting Dataset")
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=42,
        stratify=labels if len(set(labels)) > 1 else None
    )
    
    print(f"✓ Training samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
    print(f"✓ Testing samples: {len(X_test)} ({test_size*100:.0f}%)")
    
    # Show split distribution
    print(f"\n📊 Split Distribution:")
    train_dist = Counter(y_train)
    test_dist = Counter(y_test)
    
    print(f"  {'Category':<15} {'Train':<10} {'Test':<10}")
    print("  " + "-" * 35)
    for cat in sorted(set(labels)):
        print(f"  {str(cat):<15} {train_dist[cat]:<10} {test_dist[cat]:<10}")
    
    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test):
    """Advanced TF-IDF vectorization"""
    print_section("🔤 STEP 4: TF-IDF Feature Extraction")
    
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"✓ Features extracted: {X_train_tfidf.shape[1]}")
    print(f"✓ Training matrix: {X_train_tfidf.shape}")
    print(f"✓ Testing matrix: {X_test_tfidf.shape}")
    print(f"✓ Sparsity: {(1.0 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])):.2%}")
    
    # Top features
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n📌 Sample features: {', '.join(feature_names[:15])}")
    
    return X_train_tfidf, X_test_tfidf, vectorizer


def train_model(X_train_tfidf, y_train):
    """Train SVM with cross-validation"""
    print_section("🤖 STEP 5: Training SVM Model")
    
    svm_classifier = LinearSVC(
        C=1.0,
        random_state=42,
        max_iter=3000,
        dual=False
    )
    
    print("Training model...")
    svm_classifier.fit(X_train_tfidf, y_train)
    print("✓ Model training completed!")
    
    # Cross-validation
    print("\n🔄 Performing 5-Fold Cross-Validation...")
    cv_scores = cross_val_score(svm_classifier, X_train_tfidf, y_train, cv=5, scoring='accuracy')
    print(f"✓ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  Fold scores: {', '.join([f'{score:.3f}' for score in cv_scores])}")
    
    return svm_classifier


def calculate_advanced_metrics(y_test, y_pred):
    """Calculate additional advanced metrics"""
    metrics = {}
    
    # Matthews Correlation Coefficient (for binary/multiclass)
    try:
        metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
    except:
        metrics['mcc'] = None
    
    return metrics


def evaluate_model(svm_classifier, X_test_tfidf, y_test, categories):
    """Comprehensive model evaluation"""
    print_section("📈 STEP 6: Model Evaluation & Metrics")
    
    y_pred = svm_classifier.predict(X_test_tfidf)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Advanced metrics
    adv_metrics = calculate_advanced_metrics(y_test, y_pred)
    
    # Display metrics
    print("\n" + "=" * 85)
    print("  🎯 OVERALL PERFORMANCE METRICS")
    print("=" * 85)
    print()
    print(f"  {'Metric':<25} {'Score':<15} {'Percentage':<15} {'Quality':<15}")
    print("  " + "-" * 80)
    
    def get_quality(score):
        if score >= 0.9: return "Excellent ⭐⭐⭐⭐⭐"
        elif score >= 0.8: return "Very Good ⭐⭐⭐⭐"
        elif score >= 0.7: return "Good ⭐⭐⭐"
        elif score >= 0.6: return "Fair ⭐⭐"
        else: return "Needs Improvement ⭐"
    
    print(f"  {'Accuracy':<25} {accuracy:<15.4f} {accuracy*100:<15.2f}% {get_quality(accuracy)}")
    print(f"  {'Precision (weighted)':<25} {precision:<15.4f} {precision*100:<15.2f}% {get_quality(precision)}")
    print(f"  {'Recall (weighted)':<25} {recall:<15.4f} {recall*100:<15.2f}% {get_quality(recall)}")
    print(f"  {'F1-Score (weighted)':<25} {f1:<15.4f} {f1*100:<15.2f}% {get_quality(f1)}")
    
    # Macro and Micro averages
    print()
    print("  📊 Additional Averages:")
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
    print(f"  Macro F1-Score:  {macro_f1:.4f} (unweighted mean)")
    print(f"  Micro F1-Score:  {micro_f1:.4f} (total true positives)")
    
    # Classification report
    print_section("📋 Detailed Classification Report")
    print()
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    
    # Confusion matrix
    print_section("🔢 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    unique_labels = sorted(set(y_test) | set(y_pred))
    
    print()
    print("           Predicted")
    header = "         "
    for label in unique_labels:
        header += f"{str(label)[:8]:>9}"
    print(header)
    print("         " + "-" * (9 * len(unique_labels)))
    
    for i, true_label in enumerate(unique_labels):
        row = f"Actual {str(true_label)[:8]:>8}"
        for j in range(len(unique_labels)):
            row += f"{cm[i][j]:>9}"
        print(row)
    
    # Confusion matrix analysis
    print()
    print("📊 Confusion Matrix Analysis:")
    total_errors = len(y_test) - np.trace(cm)
    print(f"  Total predictions: {len(y_test)}")
    print(f"  Correct predictions: {np.trace(cm)}")
    print(f"  Incorrect predictions: {total_errors}")
    print(f"  Error rate: {(total_errors/len(y_test))*100:.2f}%")
    
    # Per-category detailed metrics
    print_section("📊 Per-Category Detailed Metrics")
    print()
    print(f"  {'Category':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10} {'Accuracy':<10}")
    print("  " + "-" * 80)
    
    for label in unique_labels:
        mask_true = y_test == label
        mask_pred = y_pred == label
        
        if mask_true.sum() > 0:
            tp = ((y_test == label) & (y_pred == label)).sum()
            fp = ((y_test != label) & (y_pred == label)).sum()
            fn = ((y_test == label) & (y_pred != label)).sum()
            tn = ((y_test != label) & (y_pred != label)).sum()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_cat = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            acc_cat = (tp + tn) / len(y_test)
            support = mask_true.sum()
            
            print(f"  {str(label):<15} {prec:<12.4f} {rec:<12.4f} {f1_cat:<12.4f} {support:<10} {acc_cat:<10.4f}")
    
    # Error analysis
    print_section("🔍 Error Analysis")
    errors = y_test != y_pred
    if errors.sum() > 0:
        print(f"\n✓ Total errors: {errors.sum()} out of {len(y_test)} predictions")
        print(f"✓ Error rate: {(errors.sum()/len(y_test))*100:.2f}%")
        
        # Most confused pairs
        print("\n📉 Most Confused Category Pairs:")
        confusion_pairs = []
        for i, true_label in enumerate(unique_labels):
            for j, pred_label in enumerate(unique_labels):
                if i != j and cm[i][j] > 0:
                    confusion_pairs.append((true_label, pred_label, cm[i][j]))
        
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        for true_cat, pred_cat, count in confusion_pairs[:5]:
            print(f"  {true_cat} → {pred_cat}: {count} times")
    else:
        print("\n🎉 Perfect classification! No errors!")
    
    return accuracy, precision, recall, f1, y_pred, adv_metrics


def display_predictions(X_test, y_test, y_pred, n_samples=10):
    """Display sample predictions with confidence"""
    print_section("🔍 Sample Predictions")
    
    print()
    n_samples = min(n_samples, len(X_test))
    
    correct = 0
    for i in range(n_samples):
        status = "✓" if y_test[i] == y_pred[i] else "✗"
        if y_test[i] == y_pred[i]:
            correct += 1
        
        print(f"{status} Text: {X_test[i][:70]}...")
        print(f"  Actual: {y_test[i]:<20} Predicted: {y_pred[i]}")
        print()
    
    print(f"Sample accuracy: {correct}/{n_samples} ({(correct/n_samples)*100:.1f}%)")


def save_comprehensive_results(accuracy, precision, recall, f1, adv_metrics, output_file="results_detailed.txt"):
    """Save comprehensive results"""
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TEXT CLASSIFICATION - COMPREHENSIVE RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision:          {precision:.4f} ({precision*100:.2f}%)\n")
        f.write(f"Recall:             {recall:.4f} ({recall*100:.2f}%)\n")
        f.write(f"F1-Score:           {f1:.4f} ({f1*100:.2f}%)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Generated by SVM Text Classifier (Enhanced Version)\n")
    
    print(f"\n✓ Comprehensive results saved to: {output_file}")


def main():
    """Main execution"""
    
    print_header("TEXT CLASSIFICATION USING SVM - ENHANCED VERSION")
    print("  Complete Metrics: Accuracy, Precision, Recall, F1-Score")
    print("  Advanced Analysis: Confusion Matrix, Error Analysis, Cross-Validation")
    
    if len(sys.argv) < 2:
        print("\n❌ ERROR: No dataset file provided")
        print("\nUsage:")
        print("  python svm_classifier_enhanced.py your_dataset.csv")
        print("\nCSV Format Required:")
        print("  text,label")
        print('  "your text here","category1"')
        sys.exit(1)
    
    dataset_file = sys.argv[1]
    
    # Pipeline
    df = load_dataset(dataset_file)
    texts, labels, categories = analyze_dataset(df)
    X_train, X_test, y_train, y_test = prepare_data(texts, labels)
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
    svm_classifier = train_model(X_train_tfidf, y_train)
    accuracy, precision, recall, f1, y_pred, adv_metrics = evaluate_model(
        svm_classifier, X_test_tfidf, y_test, categories
    )
    display_predictions(X_test, y_test, y_pred)
    
    # Final summary
    print_header("📊 FINAL SUMMARY")
    print()
    print(f"  Dataset: {dataset_file}")
    print(f"  Total samples: {len(texts)}")
    print(f"  Categories: {len(categories)}")
    print(f"  Training: {len(X_train)} | Testing: {len(X_test)}")
    print(f"  Features: {X_train_tfidf.shape[1]} TF-IDF features")
    print()
    print("  🎯 KEY METRICS:")
    print(f"    Accuracy:          {accuracy*100:.2f}%")
    print(f"    Precision:         {precision*100:.2f}%")
    print(f"    Recall:            {recall*100:.2f}%")
    print(f"    F1-Score:          {f1*100:.2f}%")
    print()
    
    save_comprehensive_results(accuracy, precision, recall, f1, adv_metrics)
    
    print_header("✓ CLASSIFICATION COMPLETED SUCCESSFULLY!")
    print()


if __name__ == "__main__":
    main()
