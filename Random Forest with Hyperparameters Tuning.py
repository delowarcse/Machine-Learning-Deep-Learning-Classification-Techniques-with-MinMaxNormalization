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

# Train, Predict & Evaluate Random Forest Model with tuned hyperparameters
sqp_rft = RandomForestClassifier(n_estimators=200, max_features='auto', random_state=42)
sqp_rft.fit(sqp_train_SX, sqp_train_y)

sqp_rft_predictions = sqp_rft.predict(sqp_test_SX)
meu.display_model_performance_metrics(true_labels=sqp_test_y, predicted_labels=sqp_rft_predictions, 
                                      classes=['stroke', 'control'])#classes=sqp_label_names)


# In[ ]:


# Train, Predict & Evaluate Random Forest Model with tuned hyperparameters
from sklearn.metrics import roc_curve
stp_rft = RandomForestClassifier(n_estimators=200, max_features='auto', random_state=42)
stp_rft.fit(sqp_train_SX, sqp_train_y)

#wqp_rft_predictions = wqp_rft.predict(wqp_test_SX)
y_pred_rft = stp_rft.predict(stp_test_SX)
#print(y_pred_rft)
y_pred_prob_rft = stp_rft.predict_proba(stp_test_SX)[:,1]

fpr_rft, tpr_rft, thresholds_rft = roc_curve(stp_test_ey, y_pred_prob_rft)

# AUC value can also be calculated like this.
from sklearn.metrics import auc
auc_rft = auc(fpr_rft, tpr_rft)
#print(auc_rft)

fig = plt.figure(1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rft, tpr_rft, label='RFT (area = {:.2f})'.format(auc_rft))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
#fig.savefig('RFT ROC Curve.png', bbox_inches='tight')
