FROM pytorch/pytorch:latest
# Set the working directory in the container
WORKDIR /app
COPY . /app
RUN apt-get update 
# RUN apt-get install -y libgl1-mesa-glx
# RUN apt-get install -yq libgtk2.0-dev
# RUN pip install -r requirements.txt
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip3 install -r requirements.txt
# Specify the default command to run when the container starts
CMD ["python", "main.py"]
#docker run ml-project python main.py --arg1 value1 --arg2 value2