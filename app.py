from flask import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("conv.csv")

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    chat = ""
    if request.method == "POST":
        old_chat = request.form["chat"]
        qts = request.form["qts"]
        qts = qts.strip().lower()
        texts = [qts] + data["question"].str.lower().tolist()
        cv = CountVectorizer()
        vector = cv.fit_transform(texts)   # pura corpus vectorize ho gaya
        cs = cosine_similarity(vector)     # so that we can find similarity
        score = cs[0][1:]
        data["score"] = score * 100
        result = data.sort_values(by="score", ascending=False)  # result sort from highest to lowest
        result = result[result.score > 10]  # woh sab result rahenge jisko score > 10

        if len(result) == 0:
            msg = "chitty --> sorry i dont know please contact - 9702400741 "
        else:
            # highest wala
            ans = result.head(1)["answer"].values[0]
            msg = "chitty --> " + (ans)

        new_chat = "you said --> " + qts + "\n" + msg
        chat = old_chat + "\n" + new_chat
        return render_template("home.html", msg=msg, chat=chat.strip())
    else:
        return render_template("home.html")
#app.run(debug=True, use_reloader=True)