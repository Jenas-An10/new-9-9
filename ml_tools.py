import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_data():
    """Load and preprocess the data."""
    df = pd.read_csv('feature_set.csv').fillna(method='pad')
    with open("negative.lncRNA.glist.xls", "r") as f:
        raw_negative_list = list(map(lambda x: x.strip(), f.readlines()))

    with open("positive.lncRNA.glist.xls", "r") as f:
        raw_positive_list = list(map(lambda x: x.strip(), f.readlines()))

    if len(raw_negative_list) < len(raw_positive_list):
        positive_list = list(np.random.choice(raw_positive_list, 150))
        negative_list = raw_negative_list
    else:
        negative_list = list(np.random.choice(raw_negative_list, 150))
        positive_list = raw_positive_list
    
    positive_df = df.loc[df['Gene_ID'].isin(positive_list)]
    negative_df = df.loc[df['Gene_ID'].isin(negative_list)]
    positive_df['label'] = 1
    negative_df['label'] = -1

    new_df = pd.concat([positive_df, negative_df])
    cols = list(new_df.columns)
    new_cols = cols[:-1]
    new_cols.insert(0, 'label')
    new_df = new_df.loc[:, new_cols]

    # Feature selection
    new_df = feature_select(new_df)
    y = np.array(new_df.label)
    X = new_df.iloc[:, 1:]
    return X, y

def feature_select(df):
    """Select important features using ExtraTreesClassifier."""
    y = np.array(df.label)
    X = np.array(df.drop(columns=['Gene_ID']).values)

    forest = ExtraTreesClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    forest.fit(X, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    best_features = ['label']
    for f in range(X.shape[1]):
        best_features.append(df.columns[indices[f] + 1])

    n_best = len(best_features)
    df1 = df.loc[:, best_features[0:n_best]]  # 0 = label
    return df1

def feature_importance_display(clf, model_id, X):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np  # Ensure numpy is imported
    
    fig, ax = plt.subplots()
    
    try:
        if model_id == 'NB':  # Bernoulli Naive Bayes
            if hasattr(clf, 'feature_log_prob_'):
                positivelog = clf.feature_log_prob_[0, :]
                negativelog = clf.feature_log_prob_[1, :]
                odds_ratios = np.exp(positivelog - negativelog)  # Odds ratio for positive class
                importances = pd.Series(odds_ratios, index=X.columns).sort_values(ascending=False)
            else:
                raise AttributeError("Naive Bayes model does not have 'feature_log_prob_' attribute.")
                
        elif model_id == 'LR':  # Logistic Regression
            if hasattr(clf, 'coef_'):
                importances = pd.Series(np.abs(clf.coef_[0]), index=X.columns).sort_values(ascending=False)
            else:
                raise AttributeError("Logistic Regression model does not have 'coef_' attribute.")
                
        elif model_id == 'RF':  # Random Forest
            if hasattr(clf, 'feature_importances_'):
                importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
            else:
                raise AttributeError("Random Forest model does not have 'feature_importances_' attribute.")
                
        else:
            raise NotImplementedError(f"Model '{model_id}' does not support feature importance extraction.")

        importances.plot(kind='bar', ax=ax)
        ax.set_title('Feature Importances')
        plt.tight_layout()
    
    except AttributeError as e:
        print(f"Error: {e}")
        raise

    return fig

def run_ML_pipeline(report, file_path, model_id):
    """
    Runs a certain pipeline on the given training data and returns the result with the most important feature.
    """
    user_x = pd.read_csv(file_path, sep=',' if file_path.endswith('.csv') else '\t', index_col=0).fillna(method='pad')
    test_x = user_x.iloc[:, 1:]  # Exclude Gene_ID

    seed = 42
    np.random.seed(seed)

    # Get the training data
    train_X, train_y = get_data()
    column_names = list(test_x.columns.values)
    train_X = train_X[column_names]

    # Split the data into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.3, random_state=seed)

    # Model selection
    if model_id == 'NB':
        from sklearn.naive_bayes import BernoulliNB
        clf = BernoulliNB(alpha=5)
    elif model_id == 'RF':
        clf = RandomForestClassifier(n_estimators=1000, random_state=seed, n_jobs=-1)
    else:
        raise NotImplementedError(f'The model_id={model_id} is not known!')

    # Train the model
    clf.fit(train_X, train_y)

    if report == 'confusion_matrix':
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(clf, test_X, test_y, ax=ax)
        return fig

    elif report == 'roc_auc_curve':
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(clf, test_X, test_y, ax=ax)
        return fig

    elif report == 'feature_importance':
        return feature_importance_display(clf, model_id, train_X)

    elif report == 'prediction_result':
        # Make predictions
        pred_y = clf.predict(test_x)
        user_x['predicted_value'] = pred_y

        # Get feature importances from the trained model
        if hasattr(clf, 'feature_importances_'):
            feature_importances = clf.feature_importances_
            important_features = np.argmax(test_x.values * feature_importances, axis=1)

            # Map the most important feature for each gene
            most_important_feature_names = [test_x.columns[i] for i in important_features]
            user_x['most_important_feature'] = most_important_feature_names

        return user_x[['Gene_ID', 'predicted_value', 'most_important_feature']]

    else:
        raise NotImplementedError(f'The report={report} is not known!')
