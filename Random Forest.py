# Import necessary dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model_evaluation_utils as meu
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

get_ipython().run_line_magic('matplotlib', 'inline')

# Load and merge datasets # white = control; red = stroke; wine = data
stroke_data = pd.read_csv('Stroke/Injure participants.csv', delim_whitespace=False)
control_data = pd.read_csv('Healthy Control participants.csv', delim_whitespace=False)

# Data type as an attribute
stroke_data['data_type'] = 'stroke'   
control_data['data_type'] = 'control'

# merge control and stroke data
datas = pd.concat([stroke_data, control_data])
datas = datas.sample(frac=1, random_state=42).reset_index(drop=True)

# understand dataset features and values
datas.head()
#stroke_data.head()
#control_data.head()


# Prepare Training and Testing Datasets
stp_features = datas.iloc[:,:-1]
stp_feature_names = stp_features.columns
stp_class_labels = np.array(datas['data_type'])

stp_train_X, stp_test_X, stp_train_y, stp_test_y = train_test_split(stp_features, stp_class_labels, 
                                                                    test_size=0.3, random_state=42)

print(Counter(stp_train_y), Counter(stp_test_y))
print('Features:', list(stp_feature_names))


# Feature Scaling
# Define the scaler 
stp_ss = StandardScaler().fit(stp_train_X)
#stp_ss = stp_train_X

# Scale the train set
stp_train_SX = stp_ss.transform(stp_train_X)
#stp_train_SX = stp_train_X

# Scale the test set
stp_test_SX = stp_ss.transform(stp_test_X)
#stp_test_SX = stp_test_X

# Encode Response class labels
le = LabelEncoder()
le.fit(stp_train_y)
# encode wine type labels
stp_train_ey = le.transform(stp_train_y)
stp_test_ey = le.transform(stp_test_y)

# # Train, Predict & Evaluate Model using Random Forests
from sklearn.ensemble import RandomForestClassifier
# train the model
sqp_rf = RandomForestClassifier()
sqp_rf.fit(sqp_train_SX, sqp_train_y)
# predict and evaluate performance
sqp_rf_predictions = sqp_rf.predict(sqp_test_SX)
meu.display_model_performance_metrics(true_labels=sqp_test_y, predicted_labels=sqp_rf_predictions, 
                                      classes=['stroke', 'control'])#classes=sqp_label_names)



# View Model ROC Curve
meu.plot_model_roc_curve(sqp_rf, sqp_test_SX, sqp_test_y)


