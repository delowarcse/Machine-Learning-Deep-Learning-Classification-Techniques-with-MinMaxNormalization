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
#stp_ss = StandardScaler().fit(stp_train_X)
stp_ss = stp_train_X

# Scale the train set
#stp_train_SX = stp_ss.transform(stp_train_X)
stp_train_SX = stp_train_X

# Scale the test set
#stp_test_SX = stp_ss.transform(stp_test_X)
stp_test_SX = stp_test_X

# Encode Response class labels
le = LabelEncoder()
le.fit(stp_train_y)
# encode wine type labels
stp_train_ey = le.transform(stp_train_y)
stp_test_ey = le.transform(stp_test_y)

# # Build & Compile DNN Model Architecture
from keras.models import Sequential
from keras.layers import Dense

stp_dnn_model = Sequential()
stp_dnn_model.add(Dense(16, activation='relu', input_shape=(13,)))
stp_dnn_model.add(Dense(16, activation='relu'))
stp_dnn_model.add(Dense(16, activation='relu'))
stp_dnn_model.add(Dense(1, activation='sigmoid'))

stp_dnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# Train the Model
history = stp_dnn_model.fit(stp_train_SX, stp_train_ey, epochs=100, batch_size=5, 
                            shuffle=True, validation_split=0.1, verbose=1)


# Predict on Test dataset
stp_dnn_ypred = stp_dnn_model.predict_classes(stp_test_SX)
stp_dnn_ypred_prob = stp_dnn_model.predict_proba(stp_test_SX)
stp_dnn_predictions = le.inverse_transform(stp_dnn_ypred)


# Evaluate Model Performance
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
t = f.suptitle('Deep Neural Net Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epochs = list(range(1,101))
fig = plt.figure(1)
ax1.plot(epochs, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks([1,50,100])#ax1.set_xticks(epochs)
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")
#fig.savefig('DNN Accuracy Curve.png', bbox_inches='tight')

fig = plt.figure(2)
ax2.plot(epochs, history.history['loss'], label='Train Loss')
ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks([1,25,60])#ax2.set_xticks(epochs)
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
#fig.savefig('DNN Loss Curve.png', bbox_inches='tight')


meu.display_model_performance_metrics(true_labels=stp_test_y, predicted_labels=stp_dnn_predictions, 
                                      classes=['stroke', 'control'])

# https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
from sklearn.metrics import roc_curve
#y_pred_dnn = wtp_dnn_model.predict(wtp_test_X).ravel()
fpr_dnn, tpr_dnn, thresholds_dnn = roc_curve(stp_test_ey, stp_dnn_ypred_prob)

# AUC value can also be calculated like this.
from sklearn.metrics import auc
auc_dnn = auc(fpr_dnn, tpr_dnn)

fig = plt.figure(1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_dnn, tpr_dnn, label='DNN (area = {:.2f})'.format(auc_dnn))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf)) # wtp_lr, wtp_test_SX, wtp_test_y
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()
#fig.savefig('DNN ROC Curve.png', bbox_inches='tight')

# calculate precision and recall for each threshold
dnn_precision, dnn_recall, _ = precision_recall_curve(stp_test_ey, stp_dnn_ypred_prob)
# calculate scores
#lr_f1, lr_auc = f1_score(wtp_test_ey, y_pred_prob_lr), auc(lr_recall, lr_precision)
fig = plt.figure(2)
pyplot.plot(dnn_recall, dnn_precision, marker='.', label='DNN')
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
#fig.savefig('DNN Precision-Recall Curve.png', bbox_inches='tight')
