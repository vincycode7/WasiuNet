from flask import Flask
from flasgger import Swagger
from blueprints.predict_blueprint import pred_bp
from configs.config import PORT, DEBUG

app = Flask(__name__)
app.register_blueprint(pred_bp)
swagger = Swagger(app, template_file='templates/swagger.json')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)