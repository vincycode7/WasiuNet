import pytest
import subprocess
import requests
import time,warnings
import tempfile
import subprocess
from requests_html import HTMLSession
import tempfile
import subprocess
import pytest
import subprocess
import requests
import time

# from bs4 import BeautifulSoup, FeatureNotFound

# @pytest.fixture(scope='session')
# def streamlit_server():
#     with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
#         proc = subprocess.Popen(["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501","--server.enableCORS=false", "--server.headless=True"], stdout=stdout_file, stderr=stderr_file)
#         time.sleep(2)
#         yield stdout_file, stderr_file
#         proc.terminate()

# def test_server(streamlit_server):
#     response = requests.get('http://localhost:8501')
#     assert 200 == response.status_code

# @pytest.fixture
# def capsys():
#     """Return capsys fixture"""
#     import sys
#     from io import StringIO
#     sys.stdout = StringIO()
#     sys.stderr = StringIO()
#     yield sys.stdout, sys.stderr
#     sys.stdout.seek(0)
#     sys.stderr.seek(0)
#     stdout = sys.stdout.read()
#     stderr = sys.stderr.read()

# from bs4 import BeautifulSoup, FeatureNotFound

@pytest.fixture(scope='session')
def streamlit_server():
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        proc = subprocess.Popen(["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=True"], stdout=stdout_file, stderr=stderr_file)
        time.sleep(2)
        yield stdout_file, stderr_file
        proc.terminate()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read().decode()
        stderr = stderr_file.read().decode()

# @pytest.fixture
# def capsys():
#     """Return capsys fixture"""
#     import sys
#     from io import StringIO
#     sys.stdout = StringIO()
#     sys.stderr = StringIO()
#     yield sys.stdout, sys.stderr
#     sys.stdout.seek(0)
#     sys.stderr.seek(0)
#     stdout = sys.stdout.read()
#     stderr = sys.stderr.read()

def test_app_delay(streamlit_server):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    time.sleep(3)
    session = HTMLSession()
    response = session.get('http://localhost:8501')
    response.html.render()
    
    assert True == True
    
def test_server(streamlit_server):
    response = requests.get('http://localhost:8501')
    assert 200 == response.status_code
    
def test_app_nameerror(streamlit_server):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # time.sleep(1)
    session = HTMLSession()
    response = session.get('http://localhost:8501')
    response.html.render()
    stdout_file, stderr_file = streamlit_server
    stdout_file.seek(0)
    stderr_file.seek(0)
    stdout = stdout_file.read().decode()
    stderr = stderr_file.read().decode()
    assert "NameError" not in stderr
    
def test_app_syntaxerror(streamlit_server):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # time.sleep(1)
    session = HTMLSession()
    response = session.get('http://localhost:8501')
    stdout_file, stderr_file = streamlit_server
    stdout_file.seek(0)
    stderr_file.seek(0)
    stdout = stdout_file.read().decode()
    stderr = stderr_file.read().decode()
    print(stderr)
    assert "SyntaxError" not in stderr
