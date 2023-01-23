from flask import Flask
from blueprints.predict_blueprint import pred_bp

app = Flask(__name__)
app.register_blueprint(pred_bp)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=True)