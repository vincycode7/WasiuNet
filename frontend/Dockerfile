FROM python:3.10.9-slim
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Install Firefox and Selenium dependencies
RUN apt-get update && apt-get install -y wget

# Install Selenium
RUN pip3 install selenium

RUN apt-get update && apt-get install -y software-properties-common

RUN apt-get install -y firefox-esr

# Add the Mozilla Firefox release PPA
# RUN apt-add-repository -y ppa:mozillateam/firefox-next

# Update the package list and install Firefox and the geckodriver package
# RUN apt-get update && apt-get install -y firefox-esr geckodriver


# Copy the application files
COPY . /app

# Expose the Streamlit port
EXPOSE 8501

# Healthcheck for the Streamlit server
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:8501/healthz || exit 1

# Start the Streamlit server
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]