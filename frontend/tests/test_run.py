import pytest
import subprocess
import requests
import time,warnings
import tempfile
import subprocess
import tempfile
import subprocess
import pytest
import subprocess
import requests
import time
from selenium import webdriver

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
    
@pytest.fixture(scope='session')
def browser():
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)
    yield driver
    driver.close()

def test_app_delay(streamlit_server):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    time.sleep(3)
    response = requests.get('http://localhost:8501')
    assert True == True
    
def test_server(streamlit_server):
    response = requests.get('http://localhost:8501')
    assert 200 == response.status_code
    
def test_app_nameerror(streamlit_server, browser):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    browser.get('http://localhost:8501')
    time.sleep(3)
    assert "NameError" not in browser.page_source
    
def test_app_syntaxerror(streamlit_server, browser):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    browser.get('http://localhost:8501')
    time.sleep(3)
    assert "SyntaxError" not in browser.page_source
