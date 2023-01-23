from flask import Flask
from views.auth_view import AuthView

app = Flask(__name__)
AuthView(app)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)