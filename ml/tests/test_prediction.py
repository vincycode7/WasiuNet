from flask import Flask
import unittest
from flask_testing import TestCase
from prediction_controller import PredictionController
from prediction import Prediction

class TestPrediction(TestCase):
    def create_app(self):
        app = Flask(__name__)
        app.config['TESTING'] = True
        return app

    def setUp(self):
        self.controller = PredictionController()
        self.prediction = Prediction()

    def test_get_prediction(self):
        # Test with valid input data
        date = '2022-01-01'
        time = '12:00:00'
        asset = 'AAPL'
        auth_data = {'user_id': 1, 'is_admin': True}
        result = self.controller.get_prediction(date, time, asset, auth_data)
        self.assertEqual(result['status'], 'success')
        self.assertTrue('prediction' in result)
        
        # Test with invalid input data
        date = '2022-01-01'
        time = '12:00:00'
        asset = 'Invalid_Asset'
        auth_data = {'user_id': 1, 'is_admin': True}
        result = self.controller.get_prediction(date, time, asset, auth_data)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['message'], 'Invalid asset name')

        # Test with unauthorized user
        date = '2022-01-01'
        time = '12:00:00'
        asset = 'AAPL'
        auth_data = {'user_id': 1, 'is_admin': False}
        result = self.controller.get_prediction(date, time, asset, auth_data)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['message'], 'Unauthorized access')