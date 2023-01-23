from flask import Flask
from views.prediction_view import PredictionView

app = Flask(__name__)
PredictionView(app)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=True)