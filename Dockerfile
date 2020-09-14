FROM python:3.8
# set a directory for the app
WORKDIR /usr/src/app
# RUN python3 --version
RUN pip --version
# copy all the files to the container
COPY . .
RUN pip install -U pip
RUN pip install --upgrade pip
# install dependencies
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python app.py