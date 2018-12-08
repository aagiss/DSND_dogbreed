FROM heroku/heroku:16

# Install pip and opencv
RUN apt-get update && apt-get install -y python-pip libsm6

# Install dependencies
ADD ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -q -r /tmp/requirements.txt

# Add our code
ADD ./app /opt/app/
ADD ./models /opt/models/
ADD ./test_images /opt/test_images/
WORKDIR /opt/

# Expose is NOT supported by Heroku
# EXPOSE 5000

RUN python app/init.py

# Run the app.  CMD is required to run on Heroku
# $PORT is set by Heroku
CMD gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT
