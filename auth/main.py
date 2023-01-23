from flask import Flask
from app.config.config import Config
from app.utils.mongo import Mongo
from app.views.auth_view import AuthView

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
mongo = Mongo()

app.add_url_rule("/register", view_func=AuthView.register, methods=["POST"])
app.add_url_rule("/login", view_func=AuthView.login, methods=["POST"])

if __name__ == "__main__":
    app.run()