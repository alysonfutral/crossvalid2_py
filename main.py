from sklearn import svm
from sklearn.model_selection import cross_val_score

input_data = [
    [1,1],
    [1,2],
    [1,3],
    [2,1],
    [2,2],
    [2,3],
    [3,1],
    [3,2],
    [3,3]
]

output_data = [1,2,3,1,2,3,1,2,3]

model = svm.SVC()
model.fit(input_data, output_data)

score = cross_val_score(model, input_data, output_data,cv = 3)
print(score)

#The output 1. 1. 1. indicates the cross-validation scores for each fold when you use cross_val_score. In this case, we used a 3-fold cross-validation cv=3, so the cross_val_score function returns an array of three accuracy scores.
#An accuracy score of 1.0 means that the model predicted the correct output for all samples in that particular fold of the cross-validation. Since we're getting [1. 1. 1.], it suggests that the model is performing perfectly on each fold of the cross-validation.
[1, 1, 1]