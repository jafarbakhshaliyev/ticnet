import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool 
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve, 
    auc,
    roc_auc_score,
    classification_report

)
from sklearn.feature_selection import SelectFromModel
import joblib
import shap
import warnings
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings('ignore')

# CONFIG: 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# processing selection
N_RUNS = 5 # Number of runs for model training and evaluation
DROP_THRESHOLD = 0.8 # Threshold for dropping columns with high missing values
FILL_NA_VALUE = 0.0 # Value to fill NaNs in the dataset

# model selection
MODEL_NAME = 'catboost'  # Name of the model to use: 'catboost', 'xgboost', or 'random_forest'
OUTPUT_PATH='./analysis/catboost'

# feature selection
MAX_FEATURES = 100  # Maximum number of features to select
FeatureSelectionThreshold = 'median' # Threshold for feature selection in CatBoost

# base model params
LEARNING_RATE_BASE = 0.1
NUMBER_OF_ITERATIONS_BASE = 200

# final CATBOOST model params
NUMBER_OF_ITERATIONS = 400
LEARNING_RATE = 0.01
DEPTH = 5
L2_REG = 5
RANDOM_STRENGTH = 0.5
BAGGING_TEMPERATURE = 0.8
BORDER_COUNT = 64
MIN_DATA_IN_LEAF = 10  # Minimum samples per leaf
MAX_CTR_COMPLEXITY = 5  # Categorical feature complexity

# final XGBoost params
SUBSAMPLE = 0.7                  
COLSAMPLE_BYTREE = 0.7         
COLSAMPLE_BYLEVEL = 0.7        
GAMMA = 1.0                 
L1_REG = 15.0 

# final Random Forest params
MAX_SAMPLES =  0.8786648106634825  # Fraction of samples per tree
MAX_FEATURES_RF = 'log2'             # Number of features considered for split
MIN_SAMPLES_SPLIT = 10            # Minimum samples to split a node

# shap analysis
DO_SHAP = True # Whether to compute SHAP values

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def drop_missing_cols(df, threshold=0.8):
    """
    Drop columns with more than a specified threshold of missing values.
    """
    col_nan_frac = df.isna().mean()
    to_drop = col_nan_frac[col_nan_frac > threshold].index
    if len(to_drop) > 0:
        print(f"Dropping {len(to_drop)} columns with more than {threshold*100}% missing values")
        for col in to_drop:
            print(f"Column '{col}' has {col_nan_frac[col] * 100:.2f}% missing values")
        df = df.drop(columns=to_drop)

    return df


def load_data(train_csv, test_csv, drop_threshold=0.8, fill_na_value=0.0):
    """
    Load train and test data from CSV files, preprocess them, and return feature matrices and labels.
    """

    # load csv files for train and test data
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    print(f"Train data shape: {df_train.shape}, Test data shape: {df_test.shape}")

    all_train_cols = set(df_train.columns)
    all_test_cols = set(df_test.columns)
    common_cols = list(all_train_cols.intersection(all_test_cols) - {"target", "filename", "label"})

    needed_cols = common_cols + ["target", "filename"]

    df_train = df_train[needed_cols].drop_duplicates()
    df_test = df_test[needed_cols].drop_duplicates()

    # drop high-missing columns
    df_train = drop_missing_cols(df_train, threshold=drop_threshold)

    keep_cols = df_train.columns
    df_test = df_test[[col for col in df_test.columns if col in keep_cols]]
    
    # fill nan values
    df_train = df_train.fillna(fill_na_value)
    df_test = df_test.fillna(fill_na_value)

    feature_cols = [col for col in df_train.columns if col not in ["target", "filename"]]
    X_train_df = df_train[feature_cols]
    y_train = df_train["target"].values

    X_test_df = df_test[feature_cols]
    y_test = df_test["target"].values
    filenames_test = df_test["filename"].values

    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weights = dict(zip(classes, class_weights))
    print(f"Class weights: {class_weights}")

    return X_train_df, y_train, X_test_df, y_test, filenames_test, class_weights


def train_model(X_train, y_train, X_test, y_test, feature_names=None, class_weights=None, random_state=42, feature_sel_threshold='median', max_selected_features=100):
    """
    Train a CatBoost model with feature selection and return the trained model and evaluation metrics.
    """
    print("Training CatBoost model...")

    sample_weights = np.array([class_weights[y] for y in y_train]) if class_weights else None

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    if MODEL_NAME.lower() == 'catboost':
        base_model = CatBoostClassifier(
            iterations=NUMBER_OF_ITERATIONS_BASE,
            learning_rate=LEARNING_RATE_BASE,
            objective='MultiClass',       
            classes_count=3,               
            eval_metric='MultiClass',      
            verbose=False,
            random_seed=random_state
        )
        base_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    elif MODEL_NAME.lower() == 'xgboost':
        base_model = xgb.XGBClassifier(
            n_estimators=NUMBER_OF_ITERATIONS_BASE,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=random_state
            )
        base_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    elif MODEL_NAME.lower() == 'random_forest':
        base_model = RandomForestClassifier(
            n_estimators=100,
            random_state=NUMBER_OF_ITERATIONS_BASE,
            class_weight=class_weights,
            n_jobs=-1
        )
        base_model.fit(X_train_scaled, y_train)


    # Feature selection
    selector = SelectFromModel(
        base_model,
        threshold = feature_sel_threshold,
        prefit=True,
        max_features=max_selected_features
    )
    mask = selector.get_support()
    X_train_sel = X_train_scaled[:, mask]
    selected_features = feature_names[mask] if feature_names is not None else None
    print(f"Feature selection: {len(selected_features)} out of total {len(feature_names)} features selected")

    if MODEL_NAME.lower() == 'catboost':
        final_model = CatBoostClassifier(
            iterations=NUMBER_OF_ITERATIONS, 
            learning_rate=LEARNING_RATE,
            depth= DEPTH,  
            l2_leaf_reg=L2_REG, 
            random_strength=RANDOM_STRENGTH,  
            bagging_temperature=BAGGING_TEMPERATURE, 
            border_count=BORDER_COUNT,  
            min_data_in_leaf=MIN_DATA_IN_LEAF, 
            max_ctr_complexity=MAX_CTR_COMPLEXITY, 
            objective='MultiClass',
            classes_count=3,
            eval_metric='MultiClass',
            verbose=True,
            random_seed=random_state
        )
        final_model.fit(X_train_sel, y_train, sample_weight=sample_weights)

    elif MODEL_NAME.lower() == 'xgboost':

        final_model = xgb.XGBClassifier(
            n_estimators= NUMBER_OF_ITERATIONS,         
            max_depth=DEPTH,            
            learning_rate=LEARNING_RATE,       
            subsample=SUBSAMPLE,           
            colsample_bytree=COLSAMPLE_BYTREE,       
            colsample_bylevel=COLSAMPLE_BYLEVEL,     
            min_child_weight=MIN_DATA_IN_LEAF,         
            gamma=GAMMA,                  
            reg_alpha=L1_REG,             
            reg_lambda=L2_REG,
            objective='multi:softprob',          
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=random_state   
        )
        final_model.fit(X_train_sel, y_train, sample_weight=sample_weights, eval_set=[(X_train_sel, y_train)],  verbose=False)
    
    elif MODEL_NAME.lower() == 'random_forest':

        final_model = RandomForestClassifier(
            n_estimators=NUMBER_OF_ITERATIONS,          
            max_depth=DEPTH,                 
            min_samples_split=MIN_SAMPLES_SPLIT, 
            min_samples_leaf=MIN_DATA_IN_LEAF,  
            max_features=MAX_FEATURES_RF,          
            max_samples=MAX_SAMPLES,            
            bootstrap=True,              
            criterion='gini',                 
            class_weight=class_weights,
            random_state=random_state,
            n_jobs=-1
        )
        final_model.fit(X_train_sel, y_train)


    y_train_pred = final_model.predict(X_train_sel)
    train_bal_acc = balanced_accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Train metrics: Balanced Accuracy: {train_bal_acc:.4f}, F1 Score: {train_f1:.4f}, Accuracy: {train_acc:.4f}")
    print('-'*70)
    print(classification_report(y_train, y_train_pred))

    return final_model, (train_bal_acc, train_f1, train_acc), {
        'scaler': scaler,
        'mask': mask,
        'selected_features': selected_features
    }


def analyze_overall_importance(avg_shap_values, features, output_path):
    """
    Analyze overall feature importance based on SHAP values and save the results.
    avg_shap_values: List of average SHAP values for each class, shape [n_classes][n_samples, n_features]
    features: List of feature names
    output_path: Path to save the analysis results
    """
    print("Analyzing overall feature importance...")

    all_class_importance = []
    for class_shap in avg_shap_values: # avg_shap_values -> LIST [n_classes][n_samples, n_features]
        all_class_importance.append(np.mean(np.abs(class_shap), axis=0))  # average over samples, shape LIST[classes](n_features,)
    feature_importance = np.mean(all_class_importance, axis=0) # shape (n_features,)

    importance_df = pd.DataFrame({
        'feature': features,
        'importance': feature_importance,
        'mean_shap': np.mean([class_shap.mean(0) for class_shap in avg_shap_values], axis=0), 
    }).sort_values('importance', ascending=False)

    top_20 = importance_df.head(20)
    top_20.to_csv(f'{output_path}/top_20_features.csv', index=False)

    vitpose_features = top_20[top_20['feature'].str.contains('body', na=False)]
    mediapipe_features = top_20[top_20['feature'].str.contains('right_eye|left_eye|right_eyebrow|left_eyebrow|mouth_outer|mouth_inner|nose|right_cheek|left_cheek|face_silhouette', na=False)]

    with open(f'{output_path}/feature_importance_analysis.txt', 'w') as f:
        f.write("Overall Feature Importance Analysis\n")

        f.write("Top 20 Most Important Features:\n")
        f.write("-" * 60 + "\n")

        for i, (_, row) in enumerate(top_20.iterrows(), 1):
            f.write(f"{i:2d}. {row['feature']:50s} | Importance: {row['importance']:.4f}\n")

        f.write(f"\n\nBody Part Distribution in Top 20:\n")
        f.write("-" * 40 + "\n")
        f.write(f"VitPose (body) features: {len(vitpose_features)}\n")
        f.write(f"MediaPipe (face) features: {len(mediapipe_features)}\n")

def analyze_class_features(all_shap_values, features, output_path):
    """
    Analyze class-specific features based on SHAP values and save the results.
    all_shap_values: Dictionary with SHAP values for each run, shape LIST [run_index]{'values': [n_classes][n_samples, n_features], 'features': ....}
    features: List of feature names
    output_path: Path to save the analysis results
    """
    print("Analyzing class-specific features...")

    class_names = {0: "FTLB", 1: "Both", 2: "Tourette"}

    all_shap_by_class = [[], [], []] 
    for run in all_shap_values.values():
        for class_idx in range(3):
            all_shap_by_class[class_idx].append(run['values'][class_idx])  # shape LIST[class][n_runs][n_samples, n_features]

    all_shap = []
    for class_idx in range(3):
        all_shap.append(np.concatenate(all_shap_by_class[class_idx], axis=0))  # shape LIST [class][n_runs*n_samples, n_features]

    all_y_test = np.concatenate([run['y_test'] for run in all_shap_values.values()])  # shape (n_runs*n_samples,)

    tourette_mask = (all_y_test == 2) # shape (n_samples,)
    both_mask = (all_y_test == 1)
    ftlb_mask = (all_y_test == 0)

    tourette_shap = all_shap[2][tourette_mask]  # true tourette samples, tourette class SHAP shape (n_tourette_samples, n_features)
    both_shap = all_shap[1][both_mask]         
    ftlb_shap = all_shap[0][ftlb_mask]  

    tourette_mean_shap = tourette_shap.mean(0) # shape (n_features,)
    both_mean_shap = both_shap.mean(0) 
    ftlb_mean_shap = ftlb_shap.mean(0)

    # feats that distinguish classes
    tourette_vs_ftlb = tourette_mean_shap - ftlb_mean_shap  # shape (n_features,)
    tourette_vs_both = tourette_mean_shap - both_mean_shap  
    both_vs_ftlb = both_mean_shap - ftlb_mean_shap 

    class_analysis_df = pd.DataFrame({
        'feature': features,
        'tourette_mean_shap': tourette_mean_shap,
        'both_mean_shap': both_mean_shap,
        'ftlb_mean_shap': ftlb_mean_shap,
        'tourette_vs_ftlb': tourette_vs_ftlb,
        'tourette_vs_both': tourette_vs_both,
        'both_vs_ftlb': both_vs_ftlb,
        'max_abs_diff': np.maximum.reduce([np.abs(tourette_vs_ftlb), np.abs(tourette_vs_both), np.abs(both_vs_ftlb)], axis=0)
    }).sort_values('max_abs_diff', ascending=False)

    class_analysis_df.head(50).to_csv(f'{output_path}/class_specific_features.csv', index=False)

    with open(f'{output_path}/class_feature_analysis.txt', 'w') as f:
        f.write("Class-Specific Feature Analysis\n")
        f.write("-" * 60 + "\n")

        f.write("TOURETTE vs FTLB Distinguishing Features:\n")
        f.write("-" * 70 + "\n")
        f.write("Features Most Predictive of Tourette (positive difference):\n")
        tourette_vs_ftlb_pos = class_analysis_df.nlargest(10, 'tourette_vs_ftlb')
        for i, (_, row) in enumerate(tourette_vs_ftlb_pos.iterrows(), 1):
            f.write(f"{i:2d}. {row['feature']:50s} | Diff: +{row['tourette_vs_ftlb']:.4f}\n")

        f.write("\nFeatures Most Predictive of FTLB (negative difference):\n")
        tourette_vs_ftlb_neg = class_analysis_df.nsmallest(10, 'tourette_vs_ftlb')
        for i, (_, row) in enumerate(tourette_vs_ftlb_neg.iterrows(), 1):
            f.write(f"{i:2d}. {row['feature']:50s} | Diff: {row['tourette_vs_ftlb']:.4f}\n")
        
        f.write("\n\nTOURETTE vs BOTH Distinguishing Features:\n")
        f.write("-" * 70 + "\n")
        f.write("Features Most Predictive of Tourette (positive difference):\n")
        tourette_vs_both_pos = class_analysis_df.nlargest(10, 'tourette_vs_both')
        for i, (_, row) in enumerate(tourette_vs_both_pos.iterrows(), 1):
            f.write(f"{i:2d}. {row['feature']:50s} | Diff: +{row['tourette_vs_both']:.4f}\n")
        f.write("\nFeatures Most Predictive of BOTH (negative difference):\n")
        tourette_vs_both_neg = class_analysis_df.nsmallest(10, 'tourette_vs_both')
        for i, (_, row) in enumerate(tourette_vs_both_neg.iterrows(), 1):
            f.write(f"{i:2d}. {row['feature']:50s} | Diff: {row['tourette_vs_both']:.4f}\n")

        f.write("\n\nBOTH vs FTLB Distinguishing Features:\n")
        f.write("-" * 70 + "\n")
        f.write("Features Most Predictive of BOTH (positive difference):\n")
        both_vs_ftlb_pos = class_analysis_df.nlargest(10, 'both_vs_ftlb')           
        for i, (_, row) in enumerate(both_vs_ftlb_pos.iterrows(), 1):
            f.write(f"{i:2d}. {row['feature']:50s} | Diff: +{row['both_vs_ftlb']:.4f}\n")
        f.write("\nFeatures Most Predictive of FTLB (negative difference):\n")
        both_vs_ftlb_neg = class_analysis_df.nsmallest(10, 'both_vs_ftlb')
        for i, (_, row) in enumerate(both_vs_ftlb_neg.iterrows(), 1):
            f.write(f"{i:2d}. {row['feature']:50s} | Diff: {row['both_vs_ftlb']:.4f}\n")

        f.write(f"\n\nOverall Most Discriminative Features (across all class pairs):\n")
        f.write("-" * 70 + "\n")
        most_discriminative = class_analysis_df.head(15)
        for i, (_, row) in enumerate(most_discriminative.iterrows(), 1):
            f.write(f"{i:2d}. {row['feature']:50s} | Max Diff: {row['max_abs_diff']:.4f}\n")
       
    
def analyze_body_parts(avg_shap_values, features, output_path): 
    """
    Analyze body part feature importance based on SHAP values and save the results.
    avg_shap_values: List of average SHAP values for each class, shape [n_classes][n_samples, n_features]
    features: List of feature names
    output_path: Path to save the analysis results
    """
    print("Analyzing body part feature importance...")

    body_importance = defaultdict(list)

    for i, feature in enumerate(features):
        class_importances = [] # shape LIST[classes](n_features,)
        for class_shap in avg_shap_values:  # avg_shap_values -> LIST [n_classes][n_samples, n_features]
            class_importances.append(np.abs(class_shap[:, i]).mean())  # average over samples, shape LIST[classes](1,)
        importance = np.mean(class_importances) # shape (1,), averaged over classes

        if 'body' in feature:
            if any(part in feature for part in ['nose', 'eye', 'ear', 'face']):
                body_importance['head_neck'].append(importance)
            elif any(part in feature for part in ['shoulder', 'elbow', 'wrist', 'hand', 'thumb', 'finger']):
                body_importance['arms_hands'].append(importance)
            elif any(part in feature for part in ['hip', 'knee', 'ankle', 'toe', 'heel']):
                body_importance['legs_feet'].append(importance)
            else:
                body_importance['vitpose_other'].append(importance)

        elif any(k in feature for k in ['right_eye', 'left_eye', 'right_eyebrow', 'left_eyebrow', 'mouth_outer', 'mouth_inner', 'nose', 'right_cheek', 'left_cheek', 'face_silhouette']):
            body_importance['mediapipe_face'].append(importance)
        else:
            body_importance['other'].append(importance)

    body_stats = {}
    for part, importances in body_importance.items():
        if importances:
            body_stats[part] = {
                'mean_importance': np.mean(importances),
                'max_importance': np.max(importances),
                'feature_cnt': len(importances),
                'total_importance': np.sum(importances)
            }

    with open(f'{output_path}/body_part_analysis.txt', 'w') as f:
        f.write("Body Part Feature Importance Analysis\n")
        f.write("-" * 60 + "\n")

        sorted_parts = sorted(body_stats.items(), 
                    key=lambda x: x[1]['total_importance'], 
                    reverse=True)

        for part, stats in sorted_parts:
            f.write(f"{part.upper().replace('_', ' ')}:\n")
            f.write(f"  Total importance: {stats['total_importance']:.4f}\n")
            f.write(f"  Mean importance: {stats['mean_importance']:.4f}\n")
            f.write(f"  Max importance: {stats['max_importance']:.4f}\n")
            f.write(f"  Number of features: {stats['feature_cnt']}\n\n")
            f.write("\n")

def analyze_individual_samples(all_shap_values, features, output_path):
    """
    Analyze individual sample SHAP values and save detailed statistics for each sample.
    all_shap_values: Dictionary with SHAP values for each run, shape LIST [run_index]{'values': [n_classes][n_samples, n_features], 'features': ....}
    features: List of feature names
    output_path: Path to save the analysis results
    """

    all_shap_combined = []
    all_y_test = []
    all_y_pred = []
    all_y_pred_proba = []
    all_filenames = []

    for run in all_shap_values.values(): # List[n_classes][n_samples, n_features]
        run_combined_shap =  np.sum([np.abs(class_shap) for class_shap in run['values']], axis=0) # shape (n_samples, n_features), averaged over classes
        all_shap_combined.append(run_combined_shap)  # List[n_runs][n_samples, n_features]
        all_y_test.append(run['y_test'])  # List[n_runs][n_samples]
        all_y_pred.append(run['y_pred'])  
        all_y_pred_proba.append(run['y_pred_proba'])  
        if isinstance(run['filenames'], (list, tuple)):
            all_filenames.extend(run['filenames'])
        else: 
            all_filenames.extend(run['filenames'].tolist())

    all_shap = np.concatenate(all_shap_combined, axis=0)  # shape (total_samples, n_features)
    all_y_test = np.concatenate(all_y_test, axis=0)  # shape (total_samples,)
    all_y_pred = np.concatenate(all_y_pred, axis=0)  # shape (total_samples,)
    all_y_pred_proba = np.concatenate(all_y_pred_proba, axis=0)  # shape (total_samples, n_classes)

    sample_stats = []
    for i in range(len(all_shap)): # total samples
        sample_shap = np.abs(all_shap[i]).sum() # shape (n_features,)

        sample_probs = all_y_pred_proba[i]  # shape (n_classes,)
        predicted_class = all_y_pred[i]  # shape (1,)
        true_class = all_y_test[i]  # shape (1,)

        predicted_class_prob = float(sample_probs[predicted_class])
        true_class_prob = float(sample_probs[true_class])

        sample_stats.append({
            'filename': all_filenames[i],
            'true_class': int(true_class),
            'predicted_class': int(predicted_class),
            'predicted_class_prob': predicted_class_prob,
            'true_class_prob': true_class_prob,
            'total_shap_impact': float(sample_shap),
            'correct_prediction': bool(predicted_class == true_class),
        })

    sample_df = pd.DataFrame(sample_stats)
    
    sample_df['predicted_class_prob'] = pd.to_numeric(sample_df['predicted_class_prob'], errors='coerce')
    sample_df['true_class_prob'] = pd.to_numeric(sample_df['true_class_prob'], errors='coerce')
    sample_df['total_shap_impact'] = pd.to_numeric(sample_df['total_shap_impact'], errors='coerce')

    with open(f'{output_path}/individual_sample_analysis.txt', 'w') as f:
        f.write("Individual Sample Analysis\n")
        f.write("-" * 60 + "\n")

        class_names = {0: "FTLB", 1: "Both", 2: "Tourette"}

        # most impactful correctly classified samples
        correct_samples = sample_df[sample_df['correct_prediction'] == True]
        if len(correct_samples) > 0:
            most_impactful_correct = correct_samples.nlargest(5, 'total_shap_impact')

            f.write("Most Impactful Correctly Classified Samples:\n")
            f.write("-" * 80 + "\n")
            for _, row in most_impactful_correct.iterrows():
                label_name = class_names[row['true_class']]
                pred_prob = row['predicted_class_prob']
                f.write(f"{row['filename']}: {label_name}\n")
                f.write(f"  SHAP Impact: {row['total_shap_impact']:.4f}\n")


        # most confident correct predictions
        if len(correct_samples) > 0:
            most_confident_correct = correct_samples.nlargest(5, 'predicted_class_prob')
            
            f.write("Most Confident Correctly Classified Samples:\n")
            f.write("-" * 80 + "\n")
            for _, row in most_confident_correct.iterrows():
                label_name = class_names[row['true_class']]
                pred_prob = row['predicted_class_prob']
                f.write(f"{row['filename']}: {label_name}\n")
                f.write(f"  Confidence: {pred_prob:.3f}\n")
                f.write(f"  SHAP Impact: {row['total_shap_impact']:.4f}\n")


        # misclassified samples
        incorrect_samples = sample_df[sample_df['correct_prediction'] == False]
        if len(incorrect_samples) > 0:
            f.write(f"Misclassified Samples:\n")
            f.write("-" * 80 + "\n")
            for _, row in incorrect_samples.iterrows():
                true_label = class_names[row['true_class']]
                pred_label = class_names[row['predicted_class']]
                pred_prob = row['predicted_class_prob']
                true_prob = row['true_class_prob']
                
                f.write(f"{row['filename']}: True={true_label}, Predicted={pred_label}\n")
                f.write(f"  True Class Prob: {true_prob:.3f}\n")
                f.write(f"  SHAP Impact: {row['total_shap_impact']:.4f}\n")

        # low confidence correct preds (including uncertain cases)
        if len(correct_samples) > 0:
            low_confidence_correct = correct_samples.nsmallest(5, 'predicted_class_prob')
            
            f.write("Low Confidence Correctly Classified Samples (Uncertain Cases):\n")
            f.write("-" * 80 + "\n")
            for _, row in low_confidence_correct.iterrows():
                label_name = class_names[row['true_class']]
                pred_prob = row['predicted_class_prob']
                f.write(f"{row['filename']}: {label_name}\n")
                f.write(f"  Low Confidence: {pred_prob:.3f}\n")
                f.write(f"  SHAP Impact: {row['total_shap_impact']:.4f}\n")

        
        # summary stats
        f.write("\nSummary Statistics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Samples: {len(sample_df)}\n")
        f.write(f"Correct Predictions: {len(correct_samples)}\n")
        f.write(f"Incorrect Predictions: {len(incorrect_samples)}\n")
        f.write(f"Overall accuracy: {len(correct_samples) / len(sample_df):.3f}\n\n")


        # confidence statistics by class
        f.write("Confidence Statistics by Class:\n")
        f.write("-" * 40 + "\n")
        for class_idx, class_name in class_names.items():
            class_samples = sample_df[sample_df['true_class'] == class_idx]
            if len(class_samples) > 0:
                avg_confidence = class_samples['predicted_class_prob'].mean()
                avg_true_prob = class_samples['true_class_prob'].mean()
                f.write(f"{class_name}:\n")
                f.write(f"  Avg predicted class confidence: {avg_confidence:.3f}\n")
                f.write(f"  Avg true class probability: {avg_true_prob:.3f}\n")
                f.write(f"  Samples: {len(class_samples)}\n\n")


    detailed_stats = []
    for _, row in sample_df.iterrows():
        detailed_row = {
            'filename': row['filename'],
            'true_label': row['true_class'],
            'predicted_label': row['predicted_class'],
            'predicted_class_probability': row['predicted_class_prob'],
            'true_class_probability': row['true_class_prob'],
            'total_shap_impact': row['total_shap_impact'],
            'correct_prediction': row['correct_prediction']
        }
        detailed_stats.append(detailed_row)
    
    detailed_df = pd.DataFrame(detailed_stats)
    detailed_df.to_csv(f'{output_path}/individual_sample_analysis.csv', index=False)

        
def analyze_feature_interactions(model, X_test, features, output_path):
    """
    Analyze feature interactions using SHAP interaction values and save the results.
    model: Trained model
    X_test: Test feature matrix (averaged one over runs)
    features: List of feature names
    output_path: Path to save the analysis results
    """

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)  # shape [array(n_samples, n_features), array(n_samples, n_features), ...]

        if isinstance(shap_values, list):
            overall_importance = np.mean([np.abs(class_shap).mean(0) for class_shap in shap_values], axis=0) # shape (n_features,)
        else:
            print('Binary Case, fix the code.')

        
        top_n = min(15, len(features))

        top_indices = np.argsort(overall_importance)[-top_n:] # sorting in ascending order

        sample_limit = min(50, X_test.shape[0]) # Limit to 50 samples for interaction analysis

        X_sample = X_test[:sample_limit, :][:, top_indices] # shape (limit_sample, top_n_features)

        interaction_values = explainer.shap_interaction_values(X_sample)  # shape LIST[n_classes][n_samples, n_features, n_features]

        if isinstance(interaction_values, list):
            avg_interactions = np.mean([np.abs(class_interactions).mean(0) for class_interactions in interaction_values], axis=0)  # shape (n_features, n_features), averaged over samples and classes
        else:
            print('Binary Case, fix the code.')
        

        with open(f'{output_path}/feature_interactions_analysis.txt', 'w') as f:
            f.write("Feature Interactions Analysis\n")
            f.write(f"Analysis based on top {top_n} features and {sample_limit} samples\n")
            f.write("Strongest feature interactions:\n")
            f.write("-" * 80 + "\n")
            
            # Find strongest interactions
            interaction_pairs = []
            for i in range(len(top_indices)):
                for j in range(i+1, len(top_indices)):
                    strength = avg_interactions[i, j]
                    interaction_pairs.append((
                        features[top_indices[i]], 
                        features[top_indices[j]], 
                        strength
                    ))

            
            # Sort by interaction strength
            interaction_pairs.sort(key=lambda x: x[2], reverse=True)
            
            for idx, (feat1, feat2, strength) in enumerate(interaction_pairs[:10], 1):
                f.write(f"{idx:2d}. {feat1[:35]:35s} <-> {feat2[:35]:35s} | {strength:.4f}\n")
            
            f.write(f"\n\nMain effects (top {top_n} features):\n")
            f.write("-" * 50 + "\n")
            for idx in np.argsort(overall_importance[top_indices])[::-1]:
                feat_name = features[top_indices[idx]]
                importance = overall_importance[top_indices[idx]]
                f.write(f"{feat_name[:50]:50s} | {importance:.4f}\n")


    except Exception as e:
        with open(f'{output_path}/feature_interactions_analysis.txt', 'w') as f:
            f.write("=== FEATURE INTERACTION ANALYSIS ===\n\n")
            f.write(f"Interaction analysis failed with error: {str(e)}\n")
            f.write("This is often due to computational constraints with large datasets.\n")
            f.write("Consider reducing the number of features or samples for interaction analysis.\n")


def analyze_misclassifications(all_shap_values, features, output_path):
    """
    Analyze misclassifications based on SHAP values and save detailed statistics.
    all_shap_values: Dictionary with SHAP values for each run, shape LIST [run_index]{'values': [n_classes][n_samples, n_features], 'features': ....}
    features: List of feature names
    output_path: Path to save the analysis results
    """
    print("Analyzing misclassifications...")

    all_shap_combined = []
    all_y_test = []
    all_y_pred = []
    all_filenames = []
    
    for run in all_shap_values.values():  # List[n_classes][n_samples, n_features]
        stacked_shap = np.stack([np.abs(class_shap) for class_shap in run['values']], axis=0)  # shape (3, n_samples, n_features)
        run_combined_shap = np.sum(stacked_shap, axis=0)  # Sum across classes: shape (n_samples, n_features)
        all_shap_combined.append(run_combined_shap)  # List[n_runs][n_samples, n_features]
        all_y_test.append(run['y_test'])  # List[n_runs][n_samples] 
        all_y_pred.append(run['y_pred'])

        if isinstance(run['filenames'], (list, tuple)):
            all_filenames.extend(run['filenames'])
        else:
            all_filenames.extend(run['filenames'].tolist())

    
    all_shap = np.concatenate(all_shap_combined, axis=0)  # shape (total_samples, n_features)
    all_y_test = np.concatenate(all_y_test, axis=0)  # shape (total_samples,)
    all_y_pred = np.concatenate(all_y_pred, axis=0)  
    all_y_pred = all_y_pred.flatten() # shape (total_samples,)
    all_filenames = np.array(all_filenames)  # shape (total_samples,)

    misclassified_mask = (all_y_test != all_y_pred)  # shape (total_samples,)
    correctly_classified_mask = ~misclassified_mask

    class_names = {0: "FTLB", 1: "Both", 2: "Tourette"}

    with open(f'{output_path}/misclassification_analysis.txt', 'w') as f:
        f.write("Misclassification Analysis\n")
        f.write("-" * 60 + "\n")

        if np.any(misclassified_mask):

            misclassified_shap = all_shap[misclassified_mask] 
            correctly_classified_shap = all_shap[correctly_classified_mask]  

            misc_importance = misclassified_shap.mean(0) # shape (n_features,)
            correct_importance = correctly_classified_shap.mean(0) # shape (n_features,)

            importance_diff = misc_importance - correct_importance  # shape (n_features,)

            misc_analysis_df = pd.DataFrame({
                'feature': features,
                'misclassified_importance': misc_importance,
                'correctly_classified_importance': correct_importance,
                'importance_difference': importance_diff,
                'abs_difference': np.abs(importance_diff)
            }).sort_values('abs_difference', ascending=False)

            misc_analysis_df.to_csv(f'{output_path}/misclassification_features.csv', index=False)

            f.write(f"Total samples: {len(all_y_test)}\n")
            f.write(f"Misclassified: {np.sum(misclassified_mask)}\n")
            f.write(f"Correctly classified: {np.sum(correctly_classified_mask)}\n")
            f.write(f"Accuracy: {np.sum(correctly_classified_mask) / len(all_y_test):.3f}\n\n")

            f.write("Misclassification patterns by true class:\n")
            f.write("-" * 50 + "\n")
            for true_class in range(3):
                true_class_mask = (all_y_test == true_class)
                true_class_misc_mask = (misclassified_mask & true_class_mask)
                if np.any(true_class_misc_mask):
                    misc_count = np.sum(true_class_misc_mask)
                    total_count = np.sum(true_class_mask)
                    f.write(f"{class_names[true_class]}: {misc_count}/{total_count} ({misc_count/total_count:.2%}) misclassified\n")
                    
                    # Show what they were predicted as
                    pred_classes = all_y_pred[true_class_misc_mask]
                    for pred_class in range(3):
                        if pred_class != true_class:
                            pred_count = np.sum(pred_classes == pred_class)
                            if pred_count > 0:
                                f.write(f"  -> {pred_count} predicted as {class_names[pred_class]}\n")

            
            # Features that distinguish misclassified samples
            f.write(f"\n\nFeatures most different in misclassified samples:\n")
            f.write("-" * 60 + "\n")
            f.write("Features MORE important in misclassified samples:\n")
            top_misc_features = misc_analysis_df.nlargest(10, 'abs_difference')
            for i, (_, row) in enumerate(top_misc_features.iterrows(), 1):
                f.write(f"{i:2d}. {row['feature']:45s} | +{row['importance_difference']:.4f}\n")
            
            f.write("\nFeatures LESS important in misclassified samples:\n")
            bottom_misc_features = misc_analysis_df.nsmallest(10, 'abs_difference')
            for i, (_, row) in enumerate(bottom_misc_features.iterrows(), 1):
                f.write(f"{i:2d}. {row['feature']:45s} | {row['importance_difference']:.4f}\n")
            
            # List specific misclassified files
            f.write(f"\n\nSpecific misclassified samples:\n")
            f.write("-" * 40 + "\n")
            misc_files = all_filenames[misclassified_mask]
            misc_true = all_y_test[misclassified_mask]
            misc_pred = all_y_pred[misclassified_mask]
            
            for i, (filename, true_label, pred_label) in enumerate(zip(misc_files, misc_true, misc_pred)):
                f.write(f"{filename}: True={class_names[true_label]}, Pred={class_names[pred_label]}\n")
        else:
            f.write("No misclassified samples found!\n")

                            

def create_comprehensive_visualizations(avg_shap_values, X_test, features, output_path):
    """
    Create comprehensive SHAP visualizations for the model's predictions.
    avg_shap_values: List of average SHAP values for each class, shape [n_classes][n_samples, n_features]
    X_test: Test feature matrix (averaged one over runs)
    features: List of feature names
    output_path: Path to save the visualizations
    """

    class_names = ['FTLB', 'Both', 'Tourette']

    try:

        for class_idx, class_name in enumerate(class_names):
            try:
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    avg_shap_values[class_idx], 
                    X_test, 
                    feature_names=features, 
                    show=False
                )
                plt.title(f'SHAP Summary Plot for {class_name}')
                plt.tight_layout()
                plt.savefig(f'{output_path}/shap_summary_{class_name}.pdf', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"SHAP summary plot for {class_name} saved to {output_path}/shap_summary_{class_name}.pdf")

            except Exception as e:
                print(f"Failed to create SHAP summary plot for {class_name}: {str(e)}")

        try:
            overall_importance = np.mean([np.abs(class_shap).mean(0) for class_shap in avg_shap_values], axis=0) # shape (n_features,)

            top_indices = np.argsort(overall_importance)[-20:]  # Get indices of top 20 features
            top_features = features[top_indices]
            top_importance = overall_importance[top_indices]

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_importance)
            plt.yticks(range(len(top_features)), [f[:50] for f in top_features])
            plt.xlabel('Mean |SHAP value|')
            plt.title('Top 20 Most Important Features (Overall)')
            plt.tight_layout()
            plt.savefig(f'{output_path}/shap_bar_plot.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved bar plot")

        except Exception as e:
            print(f"Could not create bar plot: {e}")


        try:
            if len(X_test) > 0:

                for class_idx, class_name in enumerate(class_names):

                    plt.figure(figsize=(10, 6))
                    
                    expected_value = avg_shap_values[class_idx].mean() # shape (1,)
                    
                    # Get SHAP values for first sample
                    sample_shap = avg_shap_values[class_idx][0] # shape (n_features,)
                    sample_data = X_test[0] # shape (n_features,)
                    
                    # Create explanation object
                    explanation = shap.Explanation(
                        values=sample_shap,
                        base_values=expected_value,
                        data=sample_data,
                        feature_names=features
                    )
                    
                    shap.waterfall_plot(explanation, max_display=15, show=False)
                    plt.title(f'SHAP Waterfall - {class_name} Class (Sample 1)')
                    plt.tight_layout()
                    plt.savefig(f'{output_path}/shap_waterfall_{class_name.lower()}.pdf', 
                              dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved waterfall plot for {class_name}")
        except Exception as e:
            print(f"Could not create waterfall plots: {e}")

        try:
            for class_idx, class_name in enumerate(class_names):
                plt.figure(figsize=(12, 8))
                shap.plots.beeswarm(
                    shap.Explanation(
                        values=avg_shap_values[class_idx],
                        data=X_test,
                        feature_names=features
                    ),
                    max_display=15,
                    show=False
                )
                plt.title(f'SHAP Beeswarm - {class_name} Class')
                plt.tight_layout()
                plt.savefig(f'{output_path}/shap_beeswarm_{class_name.lower()}.pdf', 
                          dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved beeswarm plot for {class_name}")
        except Exception as e:
            print(f"Could not create beeswarm plots: {e}")


    except Exception as e:
        print(f"Error in comprehensive visualizations: {e}")



def get_top_features_summary(avg_shap_values, features):
    """
    Get top features based on average SHAP values across all classes.
    avg_shap_values: List of average SHAP values for each class, shape [n_classes][n_samples, n_features]
    features: List of feature names 
    """

    all_class_importance = []
    for class_shap in avg_shap_values:  # shape LIST[class](n_samples, n_features)
        all_class_importance.append(np.abs(class_shap).mean(0)) # shape LIST[class](n_features,)

    feature_importance = np.mean(all_class_importance, axis=0)  # shape (n_features,)

    top_feats = dict(zip(features, feature_importance))  

    return dict(sorted(top_feats.items(), key=lambda x: x[1], reverse=True)[:10])


def analyze_feature_importance(model, pipeline_info, X_test_df, y_test, filenames_test, run_index, all_shap_values=None):
    """
    Analyze feature importance using SHAP values and save the results.
    model: Trained model
    pipeline_info: Dictionary with preprocessing information (scaler, mask, selected features)
    X_test_df: Test feature DataFrame
    y_test: Test labels
    filenames_test: List of filenames corresponding to the test samples
    run_index: Current run index
    all_shap_values: Dict to store SHAP values across runs (optional)
    """

    scaler = pipeline_info['scaler']
    mask = pipeline_info['mask']
    selected_features = pipeline_info['selected_features']

    X_test_scaled = scaler.transform(X_test_df)
    X_test_sel = X_test_scaled[:, mask]

    y_pred = model.predict(X_test_sel)
    y_pred_proba = model.predict_proba(X_test_sel)

    # SHAP

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sel) # shape [array(n_samples, n_features), array(n_samples, n_features), ...] 

    print(f"SHAP values structure: {type(shap_values)}, length: {len(shap_values) if isinstance(shap_values, list) else 'single array'}")

    if isinstance(shap_values, list):
        print(f"Each class shape: {[arr.shape for arr in shap_values]}")
    
    all_shap_values[run_index] = {
        'values': shap_values, # List[n_classes][n_samples, n_features]
        'features'  : selected_features,
        'X_test': X_test_sel,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'filenames': filenames_test
    }

    if run_index == N_RUNS - 1:

        print('SHAP ANALYSIS')

        # average values over runs
        all_runs_shap = [run['values'] for run in all_shap_values.values()] #  List[n_runs][n_classes][n_samples, n_features]
        avg_shap_values = []
        for class_idx in range(3):
            class_shap_values = [run_shap[class_idx] for run_shap in all_runs_shap] # List[n_runs](n_samples, n_features)
            avg_shap_values.append(np.mean(class_shap_values, axis=0)) # shape LIST[class](n_samples, n_features), it is average over runs

        avg_X_test = np.mean([run['X_test'] for run in all_shap_values.values()], axis=0)  # shape (n_samples, n_features)

        analyze_overall_importance(avg_shap_values, selected_features, OUTPUT_PATH) # overall feature importance analysis

        analyze_class_features(all_shap_values, selected_features, OUTPUT_PATH) # class-specific feature analysis

        analyze_body_parts(avg_shap_values, selected_features, OUTPUT_PATH) # body parts analysis

        analyze_individual_samples(all_shap_values, selected_features, OUTPUT_PATH) # individual sample analysis

        analyze_feature_interactions(model, avg_X_test, selected_features, OUTPUT_PATH) # feature interactions analysis
        
        analyze_misclassifications(all_shap_values, selected_features, OUTPUT_PATH) # misclassifications analysis

        create_comprehensive_visualizations(avg_shap_values, avg_X_test, selected_features, OUTPUT_PATH) # comprehensive visualizations

        print(f"SHAP analysis completed and saved to {OUTPUT_PATH}")

        return get_top_features_summary(avg_shap_values, selected_features) # return top features summary


def evaluate_model(model, pipeline_info, X_test_df, y_test, filenames_test, run_index, all_shap_values=None, do_shap=False):

    scaler = pipeline_info['scaler']
    mask = pipeline_info['mask']
    selected_features = pipeline_info['selected_features']

    X_test_scaled = scaler.transform(X_test_df)
    X_test_sel = X_test_scaled[:, mask]

    y_pred = model.predict(X_test_sel)
    y_pred_proba = model.predict_proba(X_test_sel)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    print(f"TEST metrics: Balanced Accuracy: {bal_acc:.4f}, F1 Score: {f1:.4f}, Accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}")
    print('-'*70)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix TEST:")
    print(cm)
    print('-'*70)

    if do_shap:
        top_features = analyze_feature_importance(
            model=model,
            pipeline_info=pipeline_info,
            X_test_df=X_test_df,
            y_test=y_test,
            filenames_test=filenames_test,
            run_index=run_index,
            all_shap_values=all_shap_values
        )
    
        if run_index == N_RUNS - 1:
            print("\nAveraged Feature Analysis saved to OUTPUT PATH")
            print("Averaged SHAP visualization saved as PDF in OUTPUT PATH")
    
    return bal_acc, f1, acc, roc_auc, cm



def main():

    results = {
        'train_bal_acc': [],
        'train_f1': [],
        'train_acc': [],
        'test_bal_acc': [],
        'test_f1': [],
        'test_acc': [],
        'test_roc_auc': [],
        'test_cm': [],
    }

    all_shap_values = {}

    for run in range(N_RUNS):
        print(f"Starting run {run + 1}/{N_RUNS}")

        # Load data
        X_train_df, y_train, X_test_df, y_test, filenames_test, class_weights = load_data(
            train_csv = "train_features.csv",
            test_csv = "test_features.csv",
            drop_threshold = DROP_THRESHOLD,
            fill_na_value = FILL_NA_VALUE
        )
        feature_names = np.array(X_train_df.columns)


        model, train_metrics, pipeline_info = train_model(
                X_train_df.values, y_train,
                X_test_df.values, y_test,
                feature_names=feature_names,
                class_weights=class_weights, 
                random_state=42 + run, 
                feature_sel_threshold=FeatureSelectionThreshold, 
                max_selected_features=MAX_FEATURES
            )
        train_bal_acc, train_f1, train_acc = train_metrics

        results['train_bal_acc'].append(train_bal_acc)
        results['train_f1'].append(train_f1)
        results['train_acc'].append(train_acc)

    
        test_bal_acc, test_f1, test_acc, test_roc_auc, test_cm = evaluate_model( 
            model=model,
            pipeline_info=pipeline_info,
            X_test_df=X_test_df,
            y_test=y_test,
            filenames_test=filenames_test,
            run_index=run,
            all_shap_values=all_shap_values,
            do_shap=DO_SHAP
        )

        results['test_bal_acc'].append(test_bal_acc)
        results['test_f1'].append(test_f1)
        results['test_acc'].append(test_acc)
        results['test_roc_auc'].append(test_roc_auc)
        results['test_cm'].append(test_cm)

    # Print final averaged results
    print("\n=== FINAL RESULTS ACROSS ALL RUNS ===")
    print(f"Train Balanced Accuracy: {np.mean(results['train_bal_acc']):.3f} ± {np.std(results['train_bal_acc']):.3f}")
    print(f"Train F1: {np.mean(results['train_f1']):.3f} ± {np.std(results['train_f1']):.3f}")
    print(f"Train Accuracy: {np.mean(results['train_acc']):.3f} ± {np.std(results['train_acc']):.3f}")
    print("\nTest Metrics:")
    print(f"Balanced Accuracy: {np.mean(results['test_bal_acc']):.3f} ± {np.std(results['test_bal_acc']):.3f}")
    print(f"F1 Score: {np.mean(results['test_f1']):.3f} ± {np.std(results['test_f1']):.3f}")
    print(f"Accuracy: {np.mean(results['test_acc']):.3f} ± {np.std(results['test_acc']):.3f}")
    print(f"Test ROC-AUC: {np.mean(results['test_roc_auc']):.3f} ± {np.std(results['test_roc_auc']):.3f}")
    
    # Calculate and print averaged confusion matrix
    avg_cm = np.mean(results['test_cm'], axis=0)
    cm_std = np.std(results['test_cm'], axis=0)
    print("\nAveraged Confusion Matrix:")
    print(avg_cm)
    print("\nConfusion Matrix Standard Deviation:")
    print(cm_std)
    
    # Save detailed results to file
    with open(f'{OUTPUT_PATH}/averaged_results.txt', 'w') as f:
        
        # Training metrics
        f.write("Training Metrics:\n")
        f.write("-----------------\n")
        f.write(f"Balanced Accuracy: {np.mean(results['train_bal_acc']):.3f} ± {np.std(results['train_bal_acc']):.3f}\n")
        f.write(f"F1 Score: {np.mean(results['train_f1']):.3f} ± {np.std(results['train_f1']):.3f}\n")
        f.write(f"Accuracy: {np.mean(results['train_acc']):.3f} ± {np.std(results['train_acc']):.3f}\n\n")
        
        # Test metrics
        f.write("Test Metrics:\n")
        f.write("-------------\n")
        f.write(f"Balanced Accuracy: {np.mean(results['test_bal_acc']):.3f} ± {np.std(results['test_bal_acc']):.3f}\n")
        f.write(f"F1 Score: {np.mean(results['test_f1']):.3f} ± {np.std(results['test_f1']):.3f}\n")
        f.write(f"Accuracy: {np.mean(results['test_acc']):.3f} ± {np.std(results['test_acc']):.3f}\n\n")
        f.write(f"Test ROC-AUC: {np.mean(results['test_roc_auc']):.3f} ± {np.std(results['test_roc_auc']):.3f}\n\n")
        
        # Confusion Matrix
        f.write("Averaged Confusion Matrix:\n")
        f.write("--------------------------\n")
        f.write(str(avg_cm))
        f.write("\n\nConfusion Matrix Standard Deviation:\n")
        f.write(str(cm_std))
        
        # Per-run results
        f.write("\n\nPer-run Results:\n")
        f.write("----------------\n")
        for run in range(N_RUNS):
            f.write(f"\nRun {run + 1}:\n")
            f.write(f"Train Balanced Accuracy: {results['train_bal_acc'][run]:.3f}\n")
            f.write(f"Train F1: {results['train_f1'][run]:.3f}\n")
            f.write(f"Train Accuracy: {results['train_acc'][run]:.3f}\n")
            f.write(f"Test Balanced Accuracy: {results['test_bal_acc'][run]:.3f}\n")
            f.write(f"Test F1: {results['test_f1'][run]:.3f}\n")
            f.write(f"Test Accuracy: {results['test_acc'][run]:.3f}\n")
            f.write(f"Test ROC-AUC: {results['test_roc_auc'][run]:.3f}\n")


if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True) 
    main()