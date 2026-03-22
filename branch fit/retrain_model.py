#!/usr/bin/env python3
"""
Retrain the BranchFit model from your dataset.
This script will create model.pkl and scaler.pkl that work with your Flask app.
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

def load_and_prepare_data(csv_file):
    """Load and prepare the training data."""
    
    print(f"Loading data from {csv_file}...")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def prepare_features_and_labels(df):
    """Prepare features (question responses) and labels (branch names)."""
    
    print("\nPreparing features and labels...")
    
    # New dataset format: first column is the target (branch), remaining columns are question responses
    target_column = df.columns[0]      # First column
    feature_columns = df.columns[1:]   # All remaining columns
    
    print(f"Feature columns ({len(feature_columns)}): {list(feature_columns)}")
    print(f"Target column: {target_column}")
    
    # Extract features (X) and target (y)
    answer_map = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Neutral": 3,
        "Agree": 4,
        "Strongly agree": 5
    }
    X_df = df[feature_columns].replace(answer_map)
    X = X_df.values.astype(float)
    y = df[target_column].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Unique branches: {np.unique(y)}")
    
    return X, y, list(feature_columns), target_column

def train_model(X, y):
    """Train the Random Forest model and scaler."""
    
    print("\nTraining model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Scaler fitted")
    
    # Create and train model
    # Using more estimators than the dummy model for better performance
    model = RandomForestClassifier(
        n_estimators=100,  # More trees for better performance
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train_scaled, y_train)
    print("✓ Model trained")
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler

def create_branch_labels(unique_branches):
    """Create branch labels mapping for the app."""
    
    branch_labels = {}
    for i, branch in enumerate(sorted(unique_branches)):
        branch_labels[str(i)] = branch
    
    return branch_labels

def save_artifacts(model, scaler, branch_labels):
    """Save all artifacts needed by the Flask app."""
    
    print("\nSaving model artifacts...")
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("✓ Saved model.pkl")
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("✓ Saved scaler.pkl")
    
    # Save branch labels
    with open('branch_labels.json', 'w') as f:
        json.dump(branch_labels, f, indent=2)
    print("✓ Saved branch_labels.json")
    
    # Test loading
    try:
        with open('model.pkl', 'rb') as f:
            test_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            test_scaler = pickle.load(f)
        
        print("✓ Verified: All files load correctly")
        
        # Test prediction
        n_features = test_model.n_features_in_
        test_data = np.random.rand(1, n_features)
        scaled_data = test_scaler.transform(test_data)
        probabilities = test_model.predict_proba(scaled_data)
        
        print(f"✓ Test prediction successful: {probabilities.shape}")
        
    except Exception as e:
        print(f"✗ Error testing saved files: {e}")

def main():
    """Main training process."""
    
    print("="*60)
    print("BRANCHFIT MODEL RETRAINING")
    print("="*60)
    
    # Look for CSV files in current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in current directory.")
        print("Please add your training dataset (CSV file) to this folder.")
        print("Expected format: question responses in columns, branch name in last column")
        return
    
    print(f"Found CSV files: {csv_files}")
    
    # Use the first CSV file or ask user to specify
    if len(csv_files) == 1:
        csv_file = csv_files[0]
        print(f"Using: {csv_file}")
    else:
        print("Multiple CSV files found. Please specify which one to use:")
        for i, f in enumerate(csv_files):
            print(f"  {i+1}. {f}")
        
        try:
            choice = int(input("Enter number: ")) - 1
            csv_file = csv_files[choice]
        except (ValueError, IndexError):
            print("Invalid choice. Using first file.")
            csv_file = csv_files[0]
    
    # Load and prepare data
    df = load_and_prepare_data(csv_file)
    if df is None:
        return
    
    X, y, feature_columns, target_column = prepare_features_and_labels(df)
    
    # Train model
    model, scaler = train_model(X, y)
    
    # Create branch labels
    unique_branches = np.unique(y)
    branch_labels = create_branch_labels(unique_branches)
    
    print(f"\nBranch Labels Mapping:")
    for label, branch in branch_labels.items():
        print(f"  {label}: {branch}")
    
    # Save everything
    save_artifacts(model, scaler, branch_labels)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("Your new model files are ready:")
    print("  ✓ model.pkl")
    print("  ✓ scaler.pkl") 
    print("  ✓ branch_labels.json")
    print("\nYou can now run 'py app.py' to test your Flask application!")

def retrain_with_correct_features():
    """Retrain model to match the 60 questions in the app."""
    
    print("Retraining model to match 60 questions...")
    
    # Load original dataset
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import pickle
    import json
    
    df = pd.read_csv('balanced_dataset_augmented.csv')
    
    # New dataset format: first column is the target (branch), remaining columns are question responses
    target_column = df.columns[0]
    feature_columns = df.columns[1:]  # All 60 question columns
    
    answer_map = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Neutral": 3,
        "Agree": 4,
        "Strongly agree": 5
    }
    X_df = df[feature_columns].replace(answer_map)
    X = X_df.values.astype(float)
    y = df[target_column].values
    
    print(f"Using {X.shape[1]} features (matching question count)")
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train with better parameters for higher accuracy
    model = RandomForestClassifier(
        n_estimators=200,  # More trees
        max_depth=15,      # Deeper trees
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"New model accuracy: {accuracy:.3f}")
    
    # Save new model files
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save branch labels mapping
    branch_labels = {}
    for i, branch in enumerate(sorted(np.unique(y))):
        branch_labels[str(i)] = branch
    with open('branch_labels.json', 'w') as f:
        json.dump(branch_labels, f, indent=2)
    
    print("Model retrained and saved with 60 features")
    return accuracy

# Run the retraining
accuracy = retrain_with_correct_features()