import random

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
import pandas as pd

def Formatter(X):
    X_out = X.copy()

    if "Name" in X_out.columns:
        X_out["Title"] = (
            X_out["Name"]
            .str.extract(r',\s*([^\.]+)\.')
            .fillna("Rare")
        )

        # Optional: Group rare titles exactly like during training
        X_out["Title"] = X_out["Title"].replace(
            [
                "Lady","Countess","Capt","Col","Don","Dr","Major",
                "Rev","Sir","Jonkheer","Dona"
            ],
            "Rare"
        )

        X_out["Title"] = X_out["Title"].replace({
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs"
        })

    return X_out
model = pickle.load(open("model_pipe.pkl", "rb"))

print("Model expects:", model.n_features_in_)

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    survival_prob = None
    death_prob = None
    passenger = None

    if request.method == "POST":

        try:
            passengerid = random.randint(1, 1000)

            pclass = int(request.form["pclass"])
            name = request.form["name"]
            sex = request.form["sex"]
            age = float(request.form["age"])
            sibsp = int(request.form["sibsp"])
            parch = int(request.form["parch"])
            ticket = request.form["ticket"]
            fare = float(request.form["fare"])
            cabin = request.form["cabin"]
            embarked = request.form["embarked"]

            # Create DataFrame (recommended for sklearn pipelines)
            features = pd.DataFrame([{
                "PassengerId": passengerid,
                "Pclass": pclass,
                "Name": name,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Ticket": ticket,
                "Fare": fare,
                "Cabin": cabin,
                "Embarked": embarked
            }])

            print(features)

            pred = model.predict(features)[0]

            # Probability
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                death_prob = round(probs[0] * 100, 2)
                survival_prob = round(probs[1] * 100, 2)

            prediction = (
                " Passenger is likely to survive"
                if pred == 1
                else " Passenger is unlikely to survive"
            )

            passenger = {
                "name": name,
                "age": age,
                "sex": sex,
                "class": pclass,
                "fare": fare,
                "embarked": embarked,
                "cabin": cabin
            }

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        survival_prob=survival_prob,
        death_prob=death_prob,
        passenger=passenger
    )
# pipeline visualization 

@app.route("/pipeline")
def pipeline():
    return render_template("pipeline.html")



# Print what the model expects
if hasattr(model, "feature_names_in_"):
    print("Expected columns:", model.feature_names_in_)


# print("App columns:", df.columns)
if __name__ == "__main__":
    app.run(debug=True)