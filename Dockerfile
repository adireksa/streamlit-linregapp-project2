FROM python:3.8
VOLUME /
EXPOSE 8501
WORKDIR /streamlit_server
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamline run app.py
