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
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN dpkg -i google-chrome-stable_current_amd64.deb; apt-get -fy install

# Install ChromeDriver
# Automatically download the latest chrome driver
RUN apt-get update && apt-get install -y curl unzip

RUN wget https://chromedriver.storage.googleapis.com/109.0.5414.74/chromedriver_linux64.zip
RUN unzip chromedriver_linux64.zip
RUN mv chromedriver /usr/local/bin

RUN apt-get install -y xvfb

# Copy the application files
COPY . /app

# Expose the Streamlit port
EXPOSE 8501

# Healthcheck for the Streamlit server
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD curl --fail http://localhost:8501/healthz || exit 1

# Start the Streamlit server
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]