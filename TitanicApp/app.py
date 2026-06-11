from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

print("Model expects:", model.n_features_in_)
print("Scaler expects:", scaler.n_features_in_)

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        try:

            pclass = int(request.form["pclass"])
            age = float(request.form["age"])
            sibsp = int(request.form["sibsp"])
            parch = int(request.form["parch"])
            fare = float(request.form["fare"])

            sex = request.form["sex"]
            embarked = request.form["embarked"]
            title = request.form["title"]

            # One Hot Encoding

            sex_male = 1 if sex == "male" else 0

            embarked_q = 1 if embarked == "Q" else 0
            embarked_s = 1 if embarked == "S" else 0

            title_miss = 1 if title == "Miss" else 0
            title_mr = 1 if title == "Mr" else 0
            title_mrs = 1 if title == "Mrs" else 0
            title_rare = 1 if title == "Rare" else 0

            features = np.array([[
                pclass,
                age,
                sibsp,
                parch,
                fare,
                sex_male,
                embarked_q,
                embarked_s,
                title_miss,
                title_mr,
                title_mrs,
                title_rare
            ]])

            print("Input Shape:", features.shape)
            print("Features:", features)

            features = scaler.transform(features)

            pred = model.predict(features)[0]

            if pred == 1:
                prediction = "You are probably gonna survive"
            else:
                prediction = "You are probably gonna die lol"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)