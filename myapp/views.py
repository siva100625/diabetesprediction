from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Create your views here.
def home(request):
    return render(request, 'myapp/home.html')

def result(request):
    data = pd.read_csv(r"C:\Users\Siva\Downloads\diabetes.csv")
    X = data.drop("Outcome", axis=1)
    Y = data["Outcome"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    
    val1 = float(request.POST['pregnancies'])
    val2 = float(request.POST['glucose'])
    val3 = float(request.POST['bloodpressure'])
    val4 = float(request.POST['skinthickness'])
    val5 = float(request.POST['insulin'])
    val6 = float(request.POST['bmi'])
    val7 = float(request.POST['diabetespedigreefunction'])
    val8 = float(request.POST['age'])

    prediction = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
    
    if prediction[0] == 1:
        result = "positive"
    else:
        result = "negative"
    
    return render(request, 'myapp/result.html', {"result": result})
