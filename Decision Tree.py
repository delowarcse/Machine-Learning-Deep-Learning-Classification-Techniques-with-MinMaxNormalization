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

# ### Train, Predict & Evaluate Model using Decision Tree

# In[22]:


# https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
# Train, Predict & Evaluate Model using Decision Tree

from sklearn.tree import DecisionTreeClassifier

stp_dt = DecisionTreeClassifier(max_depth=4) # max_depth=4 gives us best result
stp_dt.fit(sqp_train_SX, sqp_train_y)

y_pred_dt = stp_dt.predict(stp_test_SX)
y_pred_prob_dt = stp_dt.predict_proba(stp_test_SX)[:,1]

fpr_dt, tpr_dt, thresholds_dt = roc_curve(stp_test_ey, y_pred_prob_dt)

# AUC value can also be calculated like this.
from sklearn.metrics import auc
auc_dt = auc(fpr_dt, tpr_dt)
#print(auc_dt)


plt.xlim(0, 1)
plt.ylim(0, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_dt, tpr_dt, label='DT (area = {:.2f})'.format(auc_dt))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()
#fig.savefig('DT ROC Curve.png', bbox_inches='tight')

# calculate precision and recall for each threshold
dt_precision, dt_recall, _ = precision_recall_curve(stp_test_ey, y_pred_prob_dt)
# calculate scores
#lr_f1, lr_auc = f1_score(wtp_test_ey, y_pred_prob_lr), auc(lr_recall, lr_precision)
fig = plt.figure(2)
pyplot.plot(dt_recall, dt_precision, marker='.', label='DT')
# summarize scores
#print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# axis labels
pyplot.title('Precision-Recall Curve')
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
#fig.savefig('DT Precision-Recall Curve.png', bbox_inches='tight')



#### Train, Predict & Evaluate Model using Decision Tree

from sklearn.tree import DecisionTreeClassifier

sqp_dt = DecisionTreeClassifier(max_depth=4)
sqp_dt.fit(sqp_train_SX, sqp_train_y)

sqp_dt_predictions = sqp_dt.predict(sqp_test_SX)

meu.display_model_performance_metrics(true_labels=sqp_test_y, predicted_labels=sqp_dt_predictions, 
                                      classes=['stroke', 'control'])#classes=stp_class_labels)


# View Model ROC Curve
meu.plot_model_roc_curve(sqp_dt, sqp_test_SX, sqp_test_y)



# View Feature Importances from Decision Tree Model
sqp_dt_feature_importances = sqp_dt.feature_importances_
sqp_dt_feature_names, sqp_dt_feature_scores = zip(*sorted(zip(sqp_feature_names, sqp_dt_feature_importances), 
                                                          key=lambda x: x[1]))
y_position = list(range(len(sqp_dt_feature_names)))
fig = plt.figure(1)
plt.barh(y_position, sqp_dt_feature_scores, height=0.6, align='center')
plt.yticks(y_position , sqp_dt_feature_names)
plt.xlabel('Relative Importance Score')
plt.ylabel('Feature')
t = plt.title('Feature Importances for Decision Tree')
#fig.savefig('DT Feature Importance.png', bbox_inches='tight')


# Visualize the Decision Tree

from graphviz import Source
from sklearn import tree
from IPython.display import Image

graph = Source(tree.export_graphviz(sqp_dt, out_file=None, class_names=stp_class_labels,
                                    filled=True, rounded=True, special_characters=False,
                                    feature_names=sqp_feature_names, max_depth=4))
png_data = graph.pipe(format='png')
with open('dtree_structure.png','wb') as f:
    f.write(png_data)

Image(png_data)
