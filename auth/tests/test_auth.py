import json
import pytest
from flask import Flask
from flask_restful import Api
from views.auth_view import AuthView

@pytest.fixture
def client():
    app = Flask(__name__)
    AuthView(app)
    client = app.test_client()
    yield client

def test_get_token(client):
    response = client.post('/auth/token', json={'username': 'test', 'password': 'test'})
    data = json.loads(response.get_data(as_text=True))
    assert response.status_code == 200
    assert 'token' in data

def test_verify_token(client):
    response = client.post('/auth/token', json={'username': 'test', 'password': 'test'})
    data = json.loads(response.get_data(as_text=True))
    token = data['token']
    headers = {'Authorization': 'Bearer ' + token}
    response = client.post('/auth/verify', headers=headers)
    data = json.loads(response.get_data(as_text=True))
    assert response.status_code == 200
    assert data['message'] == 'Token is valid.'

def test_refresh_token(client):
    response = client.post('/auth/token', json={'username': 'test', 'password': 'test'})
    data = json.loads(response.get_data(as_text=True))
    token = data['token']
    headers = {'Authorization': 'Bearer ' + token}
    response = client.post('/auth/refresh', headers=headers)
    data = json.loads(response.get_data(as_text=True))
    assert response.status_code == 200
    assert 'token' in data

def test_revoke_token(client):
    response = client.post('/auth/token', json={'username': 'test', 'password': 'test'})
    data = json.loads(response.get_data(as_text=True))
    token = data['token']
    headers = {'Authorization': 'Bearer ' + token}
    response = client.post('/auth/revoke', headers=headers)
    data = json.loads(response.get_data(as_text=True))
    assert response.status_code == 200
    assert data['message'] == 'Token has been revoked.'
