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

# Implementation of Logistic Regression Methods
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from matplotlib import pyplot
import matplotlib.pyplot as plt

stp_lr = LogisticRegression()
stp_lr.fit(stp_train_SX, stp_train_y)

y_pred_lr = stp_lr.predict(stp_test_SX)
y_pred_prob_lr = stp_lr.predict_proba(stp_test_SX)[:,1]

fpr_lr, tpr_lr, thresholds_lr = roc_curve(stp_test_ey, y_pred_prob_lr)

# AUC value can also be calculated like this.
auc_lr = auc(fpr_lr, tpr_lr)
#print(auc_lr)

fig = plt.figure(1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_lr, tpr_lr, label='LR (area = {:.2f})'.format(auc_lr))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf)) # wtp_lr, wtp_test_SX, wtp_test_y
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Linear Regression ROC Curve')
plt.legend(loc='best')
plt.show()
#fig.savefig('LR ROC Curve.png', bbox_inches='tight')

# calculate precision and recall for each threshold
lr_precision, lr_recall, _ = precision_recall_curve(stp_test_ey, y_pred_prob_lr)
# calculate scores
#lr_f1, lr_auc = f1_score(wtp_test_ey, y_pred_prob_lr), auc(lr_recall, lr_precision)
plt.figure(2)
pyplot.plot(lr_recall, lr_precision, marker='.', label='LR')
# summarize scores
#print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# axis labels
fig = plt.figure(2)
pyplot.title('Precision-Recall Curve')
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
#fig.savefig('LR Precision-Recall Curve.png', bbox_inches='tight')


# Predict and Evaluate Model Performance
stp_lr_predictions = stp_lr.predict(stp_test_SX)
#print(wtp_lr_predictions)
meu.display_model_performance_metrics(true_labels=stp_test_y, predicted_labels=stp_lr_predictions, 
                                      classes=['stroke', 'control'])



# View model ROC curve
meu.plot_model_roc_curve(stp_lr, stp_test_SX, stp_test_y)

# Model Interpretation
# View Feature importances
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

stp_interpreter = Interpretation(stp_test_SX, feature_names=stp_features.columns)
stp_im_model = InMemoryModel(stp_lr.predict_proba, examples=stp_train_SX, target_names=stp_lr.classes_)
plots = stp_interpreter.feature_importance.plot_feature_importance(stp_im_model, ascending=False)
plt.xlabel('Relative Importance Score')
plt.ylabel('Feature')
plt.title('Feature Importances for Logistic Regression')
fig = plt.figure(1)
#fig.savefig('LR Feature Importance.png', bbox_inches='tight')
