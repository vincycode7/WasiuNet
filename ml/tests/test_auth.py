import pytest
import requests
from subprocess import Popen

@pytest.fixture
def flask_app():
    process = Popen(["python","app.py"])
    yield
    process.terminate()
    process.wait()

def test_api_health(flask_app):
    response = requests.get("http://localhost:5000/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "UP"