FROM python:3.10.9-slim
ENV PATH="/usr/local/bin:$PATH"
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Install Firefox and Selenium dependencies
RUN apt-get update && apt-get install -y wget

# Install Selenium
RUN pip3 install selenium

RUN apt-get update && apt-get install -y software-properties-common

RUN apt-get install -y unzip
# RUN apt-get install -y firefox-esr

# Add the Mozilla Firefox release PPA
# RUN apt-add-repository -y ppa:mozillateam/firefox-next

# Install Chrome
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add
RUN apt-get update
RUN apt-get install google-chrome-stable

# Install ChromeDriver
# Automatically download the latest chrome driver
RUN apt-get update && apt-get install -y curl unzip

# Determine the version of Chrome installed
RUN chrome_version=$(chrome --version)

# Extract the version number from the output
RUN chrome_version=${chrome_version:14:6}

RUN wget https://chromedriver.storage.googleapis.com/$chrome_version/chromedriver_linux64.zip
RUN unzip chromedriver_linux64.zip
RUN mv chromedriver /usr/local/bin

RUN apt-get install -y xvfb

# Copy the application files
COPY . /app

# Expose the Streamlit port
EXPOSE 8501

# Healthcheck for the Streamlit server
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:8501/healthz || exit 1

# Start the Streamlit server
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]