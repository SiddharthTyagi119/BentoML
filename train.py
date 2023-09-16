import bentoml
from sklearn import svm
from sklearn import datasets

#load training dataset
iris=datasets.load_iris()
x,y=iris.data,iris.target

#train the model
clf=svm.SVC(gamma='scale')
clf.fit(x,y)


#save the model
saved_model= bentoml.sklearn.save_model("iris_clf",clf)
#saved_model = bentoml.sklearn.save(model=clf, name="iris_clf")

print(f"Model saved:{saved_model}")


#isis_clf:rafkyfcuqwc6qpcs