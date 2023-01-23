from flask import Flask
import unittest
from flask_testing import TestCase
from prediction_controller import PredictionController

class TestPredictionController(TestCase):
    def create_app(self):
        app = Flask(__name__)
        app.config['TESTING'] = True
        return app

    def setUp(self):
        self.controller = PredictionController()

    def test_validate_input(self):
        # Test with valid input data
        date = '2022-01-01'
        time = '12:00:00'
        asset = 'AAPL'
        auth_data = {'user_id': 1, 'is_admin': True}
        result = self.controller.validate_input(date, time, asset, auth_data)
        self.assertIsNone(result)
        
        # Test with invalid input data
        date = 'Invalid_date'
        time = '12:00:00'
        asset = 'AAPL'
        auth_data = {'user_id': 1, 'is_admin': True}
        result = self.controller.validate_input(date,
