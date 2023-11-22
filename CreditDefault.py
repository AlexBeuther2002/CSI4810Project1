import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.spatial
from scipy.spatial import distance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, make_scorer, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import gradio as gr

creditData = pd.read_csv("CreditCardClients.csv")
workdata = (pd.DataFrame(creditData))

# print(workdata.info())
#Cov Matrix
print(creditData.isnull().sum)
creditData = creditData.drop_duplicates()
cormatrix = creditData.corr()
plt.figure(figsize = (10, 8))
plt.imshow(cormatrix, cmap = 'coolwarm', interpolation = 'none', aspect = 'auto')
plt.colorbar()
plt.xticks(range(len(cormatrix.columns)), cormatrix.columns, rotation = 45)
plt.yticks(range(len(cormatrix.columns)), cormatrix.columns)
plt.show()

#Graph of class 0 and 1
count = creditData['Y'].value_counts()
values = count.index.tolist()
counts = count.tolist()
plt.bar(values, counts, color='red')
plt.title('Count of Defaulted vs Non-Defaulted')
plt.xlabel('0: Non-Default, 1: Default')
plt.ylabel('Count')
plt.xticks(values)
plt.show()


splitdata = workdata.groupby('Y')
yesDefaultdata = splitdata.get_group(1)
noDefaultdata = splitdata.get_group(0)

X = workdata.drop('Y', axis=1)
y = workdata['Y']




#-------------Logistic Regression Using VIF----------#
combined_column = workdata.iloc[:, 12:18].sum(axis=1)
workdata['Combined_Column_12_to_17'] = combined_column
X_for_vif = workdata.drop(['Y', 'Combined_Column_12_to_17'], axis=1)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X_for_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_for_vif.values, i) for i in range(X_for_vif.shape[1])]

vif_threshold = 10
# Filter features with VIF below the threshold
selected_features = vif_data[vif_data["VIF"] < vif_threshold]["Feature"]
# Update DataFrame with selected features
X_selected = X_for_vif[selected_features]
# Run VIF again on the selected features
vif_data_selected = pd.DataFrame()
vif_data_selected["Feature"] = X_selected.columns
vif_data_selected["VIF"] = [variance_inflation_factor(X_selected.values, i) for i in range(X_selected.shape[1])]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_trainS = scaler.fit_transform(X_train)
X_testS = scaler.transform(X_test)
# Create and fit the logistic regression model
class_weights = dict(zip([0, 1], X_selected.shape[0] / (2 * np.bincount(y_train))))
model = LogisticRegression(class_weight=class_weights)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy and other metrics of the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

print(f"Log Accuracy: {accuracy}")
print("Log Classification Report:")
print(classification_report_result)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X_selected, y, cv=5, scoring=make_scorer(accuracy_score))

print(f"Log Cross Val Scores {cv_scores}")
print("Avg Cross Val Score", cv_scores.mean())
print("Log Cross Val Standard Deviation", cv_scores.std())



#-----------Normal Log Regression Analysis--------#
log_regression = LogisticRegression()
log_regression.fit(X_trainS, y_train)
y_predlog = log_regression.predict(X_testS)
logacc = accuracy_score(y_test, y_predlog)
print("Logistic Regression Accuracy Score: ", logacc)
logclassr = classification_report(y_test, y_predlog)
print(logclassr)
logcv = cross_val_score(log_regression, X, y, cv=5, scoring=make_scorer(accuracy_score))
print("Cross Val Scores: ", logcv)
print("Avg Cross Val Score: ", logcv.mean())
print("Standard Deviation: ", logcv.std())

log_odds = log_regression.predict_log_proba(X_trainS)[:, 1]

# Create individual scatterplots
num_features = X_trainS.shape[1]  # Number of features

# Set up subplots
fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(8, 4 * num_features))

#Loop through each feature and create scatterplot
for i in range(num_features):
    plt.figure(figsize=(8, 4))
    plt.scatter(X_trainS[:, i], log_odds, alpha=0.5)
    plt.xlabel(f'Feature {i}')
    plt.ylabel('Log-Odds')
    plt.title(f'Scatterplot of Feature {i} vs. Log-Odds')
    plt.tight_layout()
    plt.savefig(f'scatterplot_feature_{i}.png')
    plt.close()

for i in range(num_features):
    feature_index = i
    correlation = np.corrcoef(X_trainS[:, feature_index], log_odds)[0, 1]
    print(f"Correlation coefficient for Feature {feature_index}: {correlation:.4f}")



#---------Normal KNN Classifier--------#
knnscaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors= 5)
knn_classifier.fit(X_train, y_train)
knny_pred = knn_classifier.predict(X_test)
accuracyknn = accuracy_score(y_test, knny_pred)
print(f"KNN Normal Accuracy with selected features: {accuracyknn}")
print(classification_report(y_test, knny_pred))
print(confusion_matrix(y_test, knny_pred))

cross_val_scoresknn = cross_val_score(knn_classifier, X_train, y_train, cv= 5, scoring='accuracy')


print("KNN Normal Cross-validation scores:", cross_val_scoresknn)


print(f"KNN Normal Mean accuracy: {cross_val_scoresknn.mean():.4f}")
print(f"KNN Normal Standard deviation: {cross_val_scoresknn.std():.4f}")


#---------PCA KNN Classifier----------#
knnscaler = StandardScaler()
knnxscaled = knnscaler.fit_transform(X)
pca = PCA(n_components = 0.95)
X_pca = pca.fit_transform(knnxscaled)
kX_train, kX_test, ky_train, ky_test = train_test_split(X_pca, y, test_size=0.2, random_state= 53421)
knnx_classifier = KNeighborsClassifier(n_neighbors= 5)
knnx_classifier.fit(kX_train, ky_train)
knny_pred_prob = knnx_classifier.predict_proba(kX_test)
threshold = 0.3
knny_pred_modified = (knny_pred_prob[:, 1] > threshold).astype(int)
knnacc = accuracy_score(ky_test, knny_pred_modified)
knncr = classification_report(ky_test, knny_pred_modified)
print("KNN PCA Accuracy Score: ", knnacc)
print(knncr)
#------ PCA KNN Classifier N-Fold Cross---------#
scorersknn = {'accuracy': make_scorer(accuracy_score)}
knn_score = cross_val_score(knnx_classifier, X_trainS, y_train, cv=5, scoring=scorersknn['accuracy'])
print("\nKNN PCA Classifier 5-Fold Cross Val: ", knn_score)
print(f"Mean PCA Accuracy: {knn_score.mean()}")
print(f"Standard PCA Deviation: {knn_score.std()}\n")



#---------------Random Forest Classifier--------------------#
sample = SelectFromModel(RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced'))
sample.fit(X_trainS, y_train)
print(sample.get_support())
best_features = X_trainS[:, sample.get_support()]
feature_importances = sample.estimator_.feature_importances_

sorted_idx = feature_importances.argsort()[::-1]


X_train_selected = sample.transform(X_trainS)
X_test_selected = sample.transform(X_testS)

clf_selected = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=8, min_samples_split=10, random_state=54312)
clf_selected.fit(X_train_selected, y_train)

y_pred_selected = clf_selected.predict(X_test_selected)

accuracy_selected = accuracy_score(y_test, y_pred_selected)
print(f"Accuracy with selected features: {accuracy_selected}")
print(classification_report(y_test, y_pred_selected))
print(confusion_matrix(y_test, y_pred_selected))

cross_val_scores = cross_val_score(clf_selected, X_train_selected, y_train, cv= 5, scoring='accuracy')

print("Cross-validation scores:", cross_val_scores)
print(f"Mean accuracy: {cross_val_scores.mean():.4f}")
print(f"Standard deviation: {cross_val_scores.std():.4f}")


#-------Choosing the best Random Forest Hyperparameters--------#
n_estimators_values = [50, 100, 150]
max_depth_values = [5, 8, 10]

# Create subplots
fig, axes = plt.subplots(len(n_estimators_values), len(max_depth_values), figsize=(15, 10), sharex=True, sharey=True)

# Inside the loop
for i, n_estimators in enumerate(n_estimators_values):
    for j, max_depth in enumerate(max_depth_values):
        # Train a RandomForestClassifier with the current parameters
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=54312)
        clf.fit(X_train_selected, y_train)

        # Make predictions on the test set (not training set)
        y_pred = clf.predict(X_test_selected)

        # Calculate confusion matrix using y_test
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix with counts and a different color map
        im = axes[i, j].imshow(cm, interpolation='nearest', cmap=plt.cm.RdBu)  # You can choose a different colormap
        axes[i, j].set_title(f"n_estimators={n_estimators}\nmax_depth={max_depth}")
        
        # Add counts within each cell
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                axes[i, j].text(col, row, f"{cm[row, col]:.0f}", ha="center", va="center", color="white")

        # Add labels
        classes = np.unique(y_test)
        tick_marks = np.arange(len(classes))
        axes[i, j].set_xticks(tick_marks)
        axes[i, j].set_xticklabels(classes)
        axes[i, j].set_yticks(tick_marks)
        axes[i, j].set_yticklabels(classes)
        axes[i, j].set_ylabel('True label')
        axes[i, j].set_xlabel('Predicted label')
        
plt.tight_layout()
plt.show()



# Plot the feature importance distribution
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), sorted_idx)
plt.xlabel('Importance')
plt.title('Feature Importance Distribution')
plt.show()

#-------------Generate the odds ratio Graph--------#
logit_model = sm.Logit(y_train, sm.add_constant(X_trainS))
result = logit_model.fit()

# Get odds ratios and confidence intervals
odds_ratios = np.exp(result.params)
conf_int = np.exp(result.conf_int())

# Create a DataFrame
plot_data = pd.DataFrame({'Odds Ratio': odds_ratios, 'Lower CI': conf_int[0], 'Upper CI': conf_int[1]})

# Sort DataFrame by Odds Ratio
plot_data = plot_data.sort_values(by='Odds Ratio', ascending=True)

plt.figure(figsize=(10, 6))
sns.barplot(x=plot_data.index, y='Odds Ratio', data=plot_data, color='skyblue', label='Odds Ratio')
plt.errorbar(x=plot_data.index, y=plot_data['Odds Ratio'], yerr=[plot_data['Odds Ratio'] - plot_data['Lower CI'], plot_data['Upper CI'] - plot_data['Odds Ratio']], fmt='none', color='black', capsize=5, label='95% CI')
plt.axhline(y=1, color='red', linestyle='--', label='Reference Line (No Effect)')
plt.xticks(range(len(plot_data)), plot_data.index, rotation=45, ha='right')  # Explicitly set x-axis labels
plt.title('Odds Ratios with 95% Confidence Intervals')
plt.legend()
plt.show()


#==========Visualization Methods For Independent Variables==================#

# for feature in workdata.columns[:-1]:
#     plt.figure()
#     counts = workdata[feature].value_counts().sort_index()
#     counts.plot(kind='bar', alpha=0.7)
#     plt.title(f'{feature} vs ID')
#     plt.xlabel('Unique Values')
#     plt.ylabel('Count')
#     plt.xticks(rotation = 45)
#     plt.show()

# for feature in workdata.columns[:-1]:
#     plt.boxplot(workdata[feature])
#     plt.title(f'Box Plot of {feature}')
#     plt.ylabel('Value')
#     plt.show()

#--------Use This-----------
# for feature in workdata.columns[:-1]:
#     plt.boxplot(workdata[feature], whis=[5, 95])  # Adjust the whiskers as needed
#     plt.title(f'Box Plot of {feature}')
#     plt.ylabel('Value')
#     plt.show()







#---------prediction practice data set------------
# sampleinput = pd.DataFrame({
#     'ID': [50001],
#     'X1': [70000],
#     'X2': [1],
#     'X3': [1],
#     'X4': [1],
#     'X5': [25],
#     'X6': [-2],
#     'X7': [-2],
#     'X8': [-2],
#     'X9': [-2],
#     'X10': [-2],
#     'X11': [-2],
#     'X12': [5542],
#     'X13': [4312],
#     'X14': [1112],
#     'X15': [19231],
#     'X16': [9000],
#     'X17': [3491],
#     'X18': [4312],
#     'X19': [1112],
#     'X20': [19231],
#     'X21': [9000],
#     'X22': [3491],
#     'X23': [0],
# })

# sampleinput2 = pd.DataFrame({
#     'ID': [50001],
#     'X1': [70000],
#     'X2': [2],
#     'X3': [2],
#     'X4': [2],
#     'X5': [25],
#     'X6': [2],
#     'X7': [2],
#     'X8': [2],
#     'X9': [2],
#     'X10': [2],
#     'X11': [2],
#     'X12': [5542],
#     'X13': [4312],
#     'X14': [1112],
#     'X15': [19231],
#     'X16': [9000],
#     'X17': [3491],
#     'X18': [4312],
#     'X19': [1112],
#     'X20': [19231],
#     'X21': [9000],
#     'X22': [3491],
#     'X23': [0],
# })

#----------Prediction Practice----------
# scaledinput = scaler.transform(sampleinput)
# scaledinput2 = scaler.transform(sampleinput2)

# prediction = model.predict(scaledinput)
# prediction2 = model.predict(scaledinput2)

# print("The predicted class for the sample test is ", prediction)
# print("The predicted class for the second sample test is ", prediction2)


# #Using Gradio
# with gr.Blocks() as demo:
#     ID = gr.Number(label="Create ID Number")
#     X1 = gr.Number(label="Credit Given")
#     X2 = gr.Number(label="Gender (1 = Male, 2 = Female)")
#     X3 = gr.Number(label="Education (1 = Graduate School, 2 = University, 3 = High School, 4 = Others)")
#     X4 = gr.Number(label="Marital Status (1 = Married, 2 = Single, 3 = Other)")
#     X5 = gr.Number(label="Age (Years)")
#     X6 = gr.Number(label="The repayment status in September,  2005. (The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.)")
#     X7 = gr.Number(label="The repayment status in August,  2005. (The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.)")
#     X8 = gr.Number(label="The repayment status in July,  2005. (The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.)")
#     X9 = gr.Number(label="The repayment status in June,  2005. (The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.)")
#     X10 = gr.Number(label="The repayment status in May,  2005. (The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.)")
#     X11 = gr.Number(label="The repayment status in April,  2005. (The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.)")
#     X12 = gr.Number(label="Amount of bill statement in September, 2005. ((NT dollar))")
#     X13 = gr.Number(label="Amount of bill statement in August, 2005. ((NT dollar))")
#     X14 = gr.Number(label="Amount of bill statement in July, 2005. ((NT dollar))")
#     X15 = gr.Number(label="Amount of bill statement in June, 2005. ((NT dollar))")
#     X16 = gr.Number(label="Amount of bill statement in May, 2005. ((NT dollar))")
#     X17 = gr.Number(label="Amount of bill statement in April, 2005. ((NT dollar))")
#     X18 = gr.Number(label="Amount of previous payment in September, 2005. ((NT dollar))")
#     X19 = gr.Number(label="Amount of previous payment in August, 2005. ((NT dollar))")
#     X20 = gr.Number(label="Amount of previous payment in July, 2005. ((NT dollar))")
#     X21 = gr.Number(label="Amount of previous payment in June, 2005. ((NT dollar))")
#     X22 = gr.Number(label="Amount of previous payment in May, 2005. ((NT dollar))")
#     X23 = gr.Number(label="Amount of previous payment in April, 2005. ((NT dollar))")
#     with gr.Row():
#         add_btn = gr.Button("Calculate Default Chance")
#     c = gr.Textbox(label="Did They Default?")

    
#     def makepredict(ID, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21, X22, X23):
#         data = [ID, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21, X22, X23]
#         scaledata = scaler.transform([data])
#         predictionresult = model.predict(scaledata)
#         outputlabel = "Yes, They Will Default." if predictionresult[0] == 1 else "No, They Will Not Default."
#         return outputlabel
    
#     add_btn.click(makepredict, inputs=[ID, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21, X22, X23], outputs=c)


# demo.launch(show_api=False) 

#[[X1], [X2], [X3], [X4], [X5], [X6], [X7], [X8], [X9], [X10], [X11], [X12], [X13], [X14], [X15], [X16], [X17], [X18], [X19], [X20], [X21], [X22], [X23]]