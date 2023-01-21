FROM python:3.10.9-slim
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Install Firefox and Selenium dependencies
RUN apt-get update && \
    apt-get install -y firefox && \
    apt-get install -y wget

# Install Selenium
RUN pip3 install selenium

# Download and install the geckodriver for Firefox
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.26.0/geckodriver-v0.26.0-linux64.tar.gz && \
    tar -xvzf geckodriver-v0.26.0-linux64.tar.gz && \
    chmod +x geckodriver && \
    mv geckodriver /usr/local/bin/

# Copy the application files
COPY . /app

# Expose the Streamlit port
EXPOSE 8501

# Healthcheck for the Streamlit server
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:8501/healthz || exit 1

# Start the Streamlit server
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]