from flask import Flask
from flasgger import Swagger
from blueprints.predict_blueprint import pred_bp
from configs.config import PORT, DEBUG
import logging

app = Flask(__name__)
app.register_blueprint(pred_bp)
swagger = Swagger(app, template_file='templates/swagger.json')
# Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(levelname)s %(message)s',
#                     handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)