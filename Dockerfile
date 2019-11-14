FROM python:3

# RUN pip3 install numpy pandas jupyter tensorflow scikit-image matplotlib
COPY ./requirements.txt /project/requirements.txt
WORKDIR /project
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]