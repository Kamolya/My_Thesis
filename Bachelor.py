import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tabulate import tabulate
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns

# bendakhal el excel 3al code 
csv_file_path = r'C:\Users\ZenBook\Desktop\Gam3a\8th Semester\Bachelor\insurance fraud claims.csv'
#setting el data set as df
df = pd.read_csv(csv_file_path)
#process of preaparing data 
#bakhod el numeric data bas 3ashan ashoof el correlation we el variance 3ashan asheel el hagat el mesh mohema 
numeric_data = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
#haytala3 heatmap le el correlation between the features we ely highly correalated han remove 
variances = numeric_data.var()
plt.figure(figsize=(10, 6))
variances.plot(kind='bar', color='skyblue')
plt.title('Variances of Features')
plt.xlabel('Features')
plt.ylabel('Variance')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#haytala3 vaariance graph we el data ely mokhtalefa keteer han remove bardo 
#han remove also el data ely mesh useful according to el understanding beta3 el data 
#(I recommended to leave the detailed and remove the total maybe)Dr. Ayman
# Discussed capital loss and gains, probably on claim level, might need to keep both as have different meaning, 
#and must pre-process the case when a claim has both loss and gains (maybe keep the higher one). 
hc=['total_claim_amount']
# hc= highly correlated feature 
#wont remove pollicy annual premium  which is the high variance because it is important 
irr=['policy_number','policy_bind_date','insured_zip','insured_occupation','insured_hobbies','incident_date','incident_city','incident_location','auto_model','_c39']
#irr= irrelevant 
df.drop(columns=hc, inplace=True)
df.drop(columns=irr, inplace=True)



# Extract CSL limits 3ashan ne2semhom 3ala 2 columns 3ashan heya fe format masalan 250/500
# Split 'policy_csl' column by '/' and expand it into separate columns
split_values = df['policy_csl'].str.split('/', expand=True)

# Convert the split values to integers
df['csl_per_person'] = split_values[0].astype(int)
df['csl_per_accident'] = split_values[1].astype(int)

# Drop the original 'policy_csl' column
df.drop(columns=['policy_csl'], inplace=True)

# han7awel el ? le seperate values 
# Replace "?" with "no collision" in the 'collision_type' column
df['collision_type'] = df['collision_type'].replace('?', 'no collision')

# Replace "?" with "missing" in the 'property_damage' column
df['property_damage'] = df['property_damage'].replace('?', 'missing')

# Replace "?" with "missing" in the 'police_report_available' column
df['police_report_available'] = df['police_report_available'].replace('?', 'missing')

# Compare the absolute values of capital gains and losses
abs_gains = np.abs(df['capital-gains'])
abs_losses = np.abs(df['capital-loss'])

# Check if absolute values are equal
equal_abs = abs_gains == abs_losses

# Return the larger value with its original sign or 0 if they are equal
larger_value = np.where(abs_gains > abs_losses, df['capital-gains'], np.where(abs_losses > abs_gains, df['capital-loss'], 0))

# Adjust the sign based on the original values
for i in range(len(larger_value)):
    if equal_abs[i]:
        larger_value[i] = 0
    elif larger_value[i] == df['capital-loss'].iloc[i]:
        larger_value[i] = df['capital-loss'].iloc[i]

# Add the new column 'net-gains' to the DataFrame
df['net-gains'] = larger_value

# Drop the old columns 'capital-gains' and 'capital-loss'
df.drop(columns=['capital-gains', 'capital-loss'], inplace=True)

df['fraud_reported'] = df['fraud_reported'].replace({'N': 0, 'Y': 1})
# Move 'fraud_reported' column to the end
df = df[[col for col in df.columns if col != 'fraud_reported'] + ['fraud_reported']]



# han satrt el encoding beta3 el categorical we dealing with missing values 
# List of columns to encode
columns_to_encode = ['policy_state', 'insured_sex', 'insured_education_level', 'insured_relationship', 
                     'incident_type','collision_type','incident_severity','authorities_contacted',
                     'incident_state','property_damage','police_report_available','auto_make']

# Initialize OneHotEncoder without specifying 'sparse' and 'dtype' parameters
onehot_encoder = OneHotEncoder()

# Apply one-hot encoding to selected columns
encoded_features = onehot_encoder.fit_transform(df[columns_to_encode])

# Create a DataFrame from the encoded features with appropriate column names
encoded_df = pd.DataFrame(encoded_features.toarray(), columns=onehot_encoder.get_feature_names_out(columns_to_encode))

# Drop the original columns from the DataFrame
df.drop(columns=columns_to_encode, inplace=True)

# Concatenate the encoded columns with the original DataFrame
df = pd.concat([df, encoded_df], axis=1)


print(tabulate(df, headers='keys', tablefmt='pretty'))



#naiive bayes over 
# Create a Gaussian Naive Bayes classifier
naive_bayes_classifier = GaussianNB()

# Split the data into features (X) and target (y)
X = df.drop(columns=['fraud_reported'])
y = df['fraud_reported']

# Apply SMOTE to generate synthetic samples for the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the model (e.g., Naive Bayes) on the resampled data
naive_bayes_classifier.fit(X_train, y_train)

# Perform cross-validation predictions
y_pred_cv = cross_val_predict(naive_bayes_classifier, X_resampled, y_resampled, cv=5)

# Generate classification report
report = classification_report(y_resampled, y_pred_cv)
print(report)

#naiive bayes under 

# Split the data into features (X) and target (y)
X = df.drop(columns=['fraud_reported'])
y = df['fraud_reported']

# Initialize Naive Bayes classifier
naive_bayes_classifier = GaussianNB()

# Initialize under-sampler
under_sampler = RandomUnderSampler(random_state=42)

# Perform under-sampling
X_resampled, y_resampled = under_sampler.fit_resample(X, y)

# Perform cross-validation and get the scores
nb_cv_scores = cross_val_score(naive_bayes_classifier, X_resampled, y_resampled, cv=5, scoring='accuracy')


# Perform cross-validation predictions
nb_y_pred_cv = cross_val_predict(naive_bayes_classifier, X_resampled, y_resampled, cv=5)

# Generate classification report
nb_report = classification_report(y_resampled, nb_y_pred_cv)
print("Naive Bayes Classification Report:\n", nb_report)

# random forest over


# Instantiate the Random Forest classifier
random_forest_classifier = RandomForestClassifier(random_state=42)

# Split the data into features (X) and target (y)
X = df.drop(columns=['fraud_reported'])
y = df['fraud_reported']

# Apply SMOTE to generate synthetic samples for the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Perform cross-validation predictions
y_pred_cv = cross_val_predict(random_forest_classifier, X_resampled, y_resampled, cv=5)

# Generate classification report
report = classification_report(y_resampled, y_pred_cv)
print(report)

# random forest uner 
random_forest_classifier = RandomForestClassifier(random_state=42)

# Split the data into features (X) and target (y)
X = df.drop(columns=['fraud_reported'])
y = df['fraud_reported']
# Initialize under-sampler
under_sampler = RandomUnderSampler(random_state=42)

# Perform under-sampling
X_resampled, y_resampled = under_sampler.fit_resample(X, y)
# Perform cross-validation and get the scores
rf_cv_scores = cross_val_score(random_forest_classifier, X_resampled, y_resampled, cv=5, scoring='accuracy')

# Perform cross-validation predictions
rf_y_pred_cv = cross_val_predict(random_forest_classifier, X_resampled, y_resampled, cv=5)

# Generate classification report
rf_report = classification_report(y_resampled, rf_y_pred_cv)
print("Random Forest Classification Report:\n", rf_report)

#logistic reg

# Instantiate the Logistic Regression classifier
logistic_regression_classifier = LogisticRegression(max_iter=1000, random_state=42)


# Split the data into features (X) and target (y)
X = df.drop(columns=['fraud_reported'])
y = df['fraud_reported']

# Apply SMOTE to generate synthetic samples for the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Perform cross-validation predictions
y_pred_cv = cross_val_predict(logistic_regression_classifier, X_resampled, y_resampled, cv=5)

# Generate classification report
report = classification_report(y_resampled, y_pred_cv)
print(report)

#Logistic Reg under 
# Instantiate the Logistic Regression classifier
logistic_regression_classifier = LogisticRegression(max_iter=1000, random_state=42)

# Split the data into features (X) and target (y)
X = df.drop(columns=['fraud_reported'])
y = df['fraud_reported']

# Apply RandomUnderSampler to balance the classes
under_sampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = under_sampler.fit_resample(X, y)

# Perform cross-validation and get the scores
lr_cv_scores = cross_val_score(logistic_regression_classifier, X_resampled, y_resampled, cv=5, scoring='accuracy')

# Perform cross-validation predictions
lr_y_pred_cv = cross_val_predict(logistic_regression_classifier, X_resampled, y_resampled, cv=5)

# Generate classification report
lr_report = classification_report(y_resampled, lr_y_pred_cv)
print("Logistic Regression Classification Report:\n", lr_report)
