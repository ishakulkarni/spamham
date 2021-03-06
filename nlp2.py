import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

x_test = joblib.load('x_test_spamham.joblib')
y_test = joblib.load('y_test_spamham.joblib')

loaded = joblib.load('Model_spamhamSVM.joblib')
pred = loaded.predict(x_test)

print('test acc= ',accuracy_score(pred,y_test))
print('bal test acc= ',balanced_accuracy_score(pred,y_test))
print('F1 score= ',f1_score(pred,y_test,average=None))

cm = confusion_matrix(y_test,pred)
display=ConfusionMatrixDisplay(cm).plot()
plt.figure(1)
plt.title('Spam or Ham')
classNames = ['spam','ham']
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames)
plt.yticks(tick_marks, classNames)
plt.show()