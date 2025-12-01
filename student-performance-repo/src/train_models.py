#!/usr/bin/env python3
"""Train Linear and Logistic regression models on the UCI Student Performance dataset.

Usage:
    python src/train_models.py --data data/student-mat.csv --output models
"""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_data(path):
    df = pd.read_csv(path, sep=';')
    return df

def preprocess(df):
    # target for regression: G3 (final grade)
    # target for classification: pass if G3 >= 10
    df = df.copy()
    df['pass'] = (df['G3'] >= 10).astype(int)

    # select a subset of features (numerical + a few categorical)
    numeric_feats = ['age', 'absences', 'G1', 'G2']
    categorical_feats = ['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']

    X_num = df[numeric_feats]
    X_cat = df[categorical_feats]

    # imputer for numeric
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
    ])

    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    pre = ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('cat', cat_pipe, categorical_feats)
    ])

    X = pre.fit_transform(df)
    # Build feature names (for interpretability)
    cat_cols = pre.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_feats)
    feature_names = numeric_feats + list(cat_cols)
    X_df = pd.DataFrame(X, columns=feature_names, index=df.index)

    y_reg = df['G3']
    y_clf = df['pass']
    return X_df, y_reg, y_clf, pre

def train_and_eval(X, y_reg, y_clf, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # split
    X_train, X_test, yreg_train, yreg_test, yclf_train, yclf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, yreg_train)
    pred_reg = lr.predict(X_test)
    r2 = r2_score(yreg_test, pred_reg)
    rmse = mean_squared_error(yreg_test, pred_reg, squared=False)

    print('Linear Regression — R2: {:.4f}, RMSE: {:.4f}'.format(r2, rmse))

    # Logistic Regression (use solver lbfgs)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, yclf_train)
    pred_clf = clf.predict(X_test)
    prob_clf = clf.predict_proba(X_test)[:,1]
    acc = accuracy_score(yclf_test, pred_clf)
    try:
        auc = roc_auc_score(yclf_test, prob_clf)
    except:
        auc = float('nan')

    print('Logistic Regression — Accuracy: {:.4f}, AUC: {:.4f}'.format(acc, auc))
    print('\nClassification report:\n', classification_report(yclf_test, pred_clf))
    print('Confusion matrix:\n', confusion_matrix(yclf_test, pred_clf))

    # Save models and preprocessor
    joblib.dump({'model': lr}, os.path.join(output_dir, 'linear_regression.joblib'))
    joblib.dump({'model': clf}, os.path.join(output_dir, 'logistic_regression.joblib'))
    joblib.dump({'preprocessor': pre}, os.path.join(output_dir, 'preprocessor.joblib'))
    print('Saved models to', output_dir)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to student-mat.csv')
    p.add_argument('--output', default='models', help='Output directory to save models')
    args = p.parse_args()

    df = load_data(args.data)
    X, yreg, yclf, pre = preprocess(df)
    train_and_eval(X, yreg, yclf, args.output)

if __name__ == '__main__':
    main()
