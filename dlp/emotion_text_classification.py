# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
raw_df = pd.read_csv('/content/emotion_dataset.csv')  # Assuming the correct file name

# Map emotions to numeric values
emotion_mapping = {emotion: i for i, emotion in enumerate(raw_df['Emotion'].unique())}
raw_df['target'] = raw_df['Emotion'].map(emotion_mapping)

# Preprocess text (Tokenization, stop word removal, and stemming)
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer(language='english')
english_stopwords = stopwords.words('english')

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in english_stopwords]  # Stopword removal
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    return " ".join(tokens)

raw_df['processed_text'] = raw_df['Text'].apply(preprocess_text)

# Create a Bag of Words (BoW) representation
vectorizer = CountVectorizer(lowercase=True, max_features=1000)
X = vectorizer.fit_transform(raw_df['processed_text'])
y = raw_df['target']

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000, solver='sag')
model.fit(X_train, y_train)

# Make predictions
train_preds = model.predict(X_train)
val_preds = model.predict(X_val)
test_preds = model.predict(X_test)

# Evaluation metrics for the training set
train_accuracy = accuracy_score(y_train, train_preds)
train_f1 = f1_score(y_train, train_preds, average='weighted')
train_precision = precision_score(y_train, train_preds, average='weighted')
train_recall = recall_score(y_train, train_preds, average='weighted')

# Evaluation metrics for the validation set
val_accuracy = accuracy_score(y_val, val_preds)
val_f1 = f1_score(y_val, val_preds, average='weighted')
val_precision = precision_score(y_val, val_preds, average='weighted')
val_recall = recall_score(y_val, val_preds, average='weighted')

# Evaluation metrics for the test set
test_accuracy = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds, average='weighted')
test_precision = precision_score(y_test, test_preds, average='weighted')
test_recall = recall_score(y_test, test_preds, average='weighted')

# Print out evaluation metrics
print("Training Accuracy: ", train_accuracy)
print("Training F1 Score: ", train_f1)
print("Training Precision: ", train_precision)
print("Training Recall: ", train_recall)

print("Validation Accuracy: ", val_accuracy)
print("Validation F1 Score: ", val_f1)
print("Validation Precision: ", val_precision)
print("Validation Recall: ", val_recall)

print("Test Accuracy: ", test_accuracy)
print("Test F1 Score: ", test_f1)
print("Test Precision: ", test_precision)
print("Test Recall: ", test_recall)

train_accuracy

from sklearn.metrics import classification_report
from rich.console import Console
console = Console()
console.print(classification_report(y_test, test_preds, target_names=emotion_mapping.keys()))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from rich.console import Console

# Confusion Matrix for the test set
cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_mapping.keys(), yticklabels=emotion_mapping.keys())
plt.title("Confusion Matrix for Test Set")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png") # Save the confusion matrix plot
plt.show()

# Plotting Evaluation Metrics
metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
train_scores = [train_accuracy, train_f1, train_precision, train_recall]
val_scores = [val_accuracy, val_f1, val_precision, val_recall]
test_scores = [test_accuracy, test_f1, test_precision, test_recall]

df_metrics = pd.DataFrame([train_scores, val_scores, test_scores], columns=metrics, index=['Train', 'Validation', 'Test'])

df_metrics.plot(kind='bar', figsize=(10, 6), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Evaluation Metrics')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.savefig("evaluation_metrics.png") # Save the evaluation metrics plot
plt.show()

console = Console()
console.print(classification_report(y_test, test_preds, target_names=emotion_mapping.keys()))

# Save the classification report to a text file
with open("classification_report.txt", "w") as f:
  f.write(classification_report(y_test, test_preds, target_names=emotion_mapping.keys()))

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from rich.console import Console
from rich.text import Text

# Initialize console for colorful printing
console = Console()

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    model.fit(X_train, y_train)

    # Make predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    # Evaluation metrics for the training set
    train_accuracy = accuracy_score(y_train, train_preds)
    train_f1 = f1_score(y_train, train_preds, average='weighted')
    train_precision = precision_score(y_train, train_preds, average='weighted')
    train_recall = recall_score(y_train, train_preds, average='weighted')

    # Evaluation metrics for the validation set
    val_accuracy = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average='weighted')
    val_precision = precision_score(y_val, val_preds, average='weighted')
    val_recall = recall_score(y_val, val_preds, average='weighted')

    # Evaluation metrics for the test set
    test_accuracy = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average='weighted')
    test_precision = precision_score(y_test, test_preds, average='weighted')
    test_recall = recall_score(y_test, test_preds, average='weighted')

    # Use Rich for colorful printing
    console.print(f"[bold green]{model_name} - Training Metrics:[/bold green]")
    console.print(f"Training Accuracy: [bold cyan]{train_accuracy:.4f}[/bold cyan]")
    console.print(f"Training F1 Score: [bold cyan]{train_f1:.4f}[/bold cyan]")
    console.print(f"Training Precision: [bold cyan]{train_precision:.4f}[/bold cyan]")
    console.print(f"Training Recall: [bold cyan]{train_recall:.4f}[/bold cyan]")

    console.print(f"[bold green]{model_name} - Validation Metrics:[/bold green]")
    console.print(f"Validation Accuracy: [bold yellow]{val_accuracy:.4f}[/bold yellow]")
    console.print(f"Validation F1 Score: [bold yellow]{val_f1:.4f}[/bold yellow]")
    console.print(f"Validation Precision: [bold yellow]{val_precision:.4f}[/bold yellow]")
    console.print(f"Validation Recall: [bold yellow]{val_recall:.4f}[/bold yellow]")

    console.print(f"[bold green]{model_name} - Test Metrics:[/bold green]")
    console.print(f"Test Accuracy: [bold magenta]{test_accuracy:.4f}[/bold magenta]")
    console.print(f"Test F1 Score: [bold magenta]{test_f1:.4f}[/bold magenta]")
    console.print(f"Test Precision: [bold magenta]{test_precision:.4f}[/bold magenta]")
    console.print(f"Test Recall: [bold magenta]{test_recall:.4f}[/bold magenta]")

    # Confusion Matrix for the test set
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_mapping.keys(), yticklabels=emotion_mapping.keys())
    plt.title(f"Confusion Matrix for {model_name} (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Save the confusion matrix as an image
    plt.savefig(f"{model_name}_confusion_matrix.png")
    console.print(f"[bold magenta]Confusion matrix for {model_name} saved as {model_name}_confusion_matrix.png[/bold magenta]")
    plt.close()

# Naive Bayes Model
nb_model = MultinomialNB()
evaluate_model(nb_model, X_train, y_train, X_val, y_val, X_test, y_test, "Naive Bayes Model")

# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
evaluate_model(dt_model, X_train, y_train, X_val, y_val, X_test, y_test, "Decision Tree Model")

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model(rf_model, X_train, y_train, X_val, y_val, X_test, y_test, "Random Forest Model")

from sklearn.model_selection import GridSearchCV

# Define the model
rf_model = RandomForestClassifier(random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [10, 20, 30, None],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],    # Minimum samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2', None]  # Number of features to consider for splits
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model,
                           param_grid=param_grid,
                           cv=5,  # 5-fold cross-validation
                           verbose=2,  # Show progress
                           n_jobs=-1,  # Use all available CPU cores
                           scoring='accuracy')  # Use accuracy as the evaluation metric

# Fit GridSearchCV on the training data
console.print("[bold green]Fitting GridSearchCV on the training data...[/bold green]")
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearchCV
console.print(f"\n[bold cyan]Best parameters found:[/bold cyan] {grid_search.best_params_}")

# Best model from grid search
best_rf_model = grid_search.best_estimator_

# Evaluate the tuned model
train_preds = best_rf_model.predict(X_train)
val_preds = best_rf_model.predict(X_val)
test_preds = best_rf_model.predict(X_test)

# Training metrics
console.print("\n[bold magenta]Random Forest (Tuned) - Training Metrics:[/bold magenta]")
console.print(f"[bold green]Training Accuracy:[/bold green] {accuracy_score(y_train, train_preds):.4f}")
console.print(f"[bold yellow]Training F1 Score:[/bold yellow] {f1_score(y_train, train_preds, average='weighted'):.4f}")
console.print(f"[bold red]Training Precision:[/bold red] {precision_score(y_train, train_preds, average='weighted'):.4f}")
console.print(f"[bold blue]Training Recall:[/bold blue] {recall_score(y_train, train_preds, average='weighted'):.4f}")

# Validation metrics
console.print("\n[bold magenta]Random Forest (Tuned) - Validation Metrics:[/bold magenta]")
console.print(f"[bold green]Validation Accuracy:[/bold green] {accuracy_score(y_val, val_preds):.4f}")
console.print(f"[bold yellow]Validation F1 Score:[/bold yellow] {f1_score(y_val, val_preds, average='weighted'):.4f}")
console.print(f"[bold red]Validation Precision:[/bold red] {precision_score(y_val, val_preds, average='weighted'):.4f}")
console.print(f"[bold blue]Validation Recall:[/bold blue] {recall_score(y_val, val_preds, average='weighted'):.4f}")

# Test metrics
console.print("\n[bold magenta]Random Forest (Tuned) - Test Metrics:[/bold magenta]")
console.print(f"[bold green]Test Accuracy:[/bold green] {accuracy_score(y_test, test_preds):.4f}")
console.print(f"[bold yellow]Test F1 Score:[/bold yellow] {f1_score(y_test, test_preds, average='weighted'):.4f}")
console.print(f"[bold red]Test Precision:[/bold red] {precision_score(y_test, test_preds, average='weighted'):.4f}")
console.print(f"[bold blue]Test Recall:[/bold blue] {recall_score(y_test, test_preds, average='weighted'):.4f}")

# Confusion Matrix for the test set
cm = confusion_matrix(y_test, test_preds)

# Plot confusion matrix with seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_mapping.keys(), yticklabels=emotion_mapping.keys())
plt.title(f"Confusion Matrix for Random Forest (Tuned) (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig('Random_Forest_Confusion_Matrix.png')
plt.show()

console.print("\n[bold green]Training complete and evaluation metrics printed![/bold green]")

