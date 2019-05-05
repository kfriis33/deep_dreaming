from flask import Flask, render_template, request, Response
from flask_mail import Mail, Message

from generate_dream import processImage
import inception5h

from PIL import Image
import io
import os

import threading
import json
import numpy as np

# loading inception model
inception5h.maybe_download()
model = inception5h.Inception5h()

# initializing Flask application
app = Flask(__name__)

# setting up our mail configuration
mail_settings = {
    "MAIL_SERVER": 'smtp.gmail.com',
    "MAIL_PORT": 465,
    "MAIL_USE_TLS": False,
    "MAIL_USE_SSL": True,
    "MAIL_USERNAME": 'RocketAIcs01430project@gmail.com',
    "MAIL_PASSWORD": 'CS01430Password1!#'
}

app.config.update(mail_settings)

# initializing our Mail application
mail = Mail(app)

def initiate_image_processing(image, email_address, optimization_layer, num_iterations, num_repetitions, blend):
	with app.app_context(): # throwing this into a new app context
		# getting our resulting image(s)
		print("initating image processing and email construction")
		results = processImage(model,image, email_address, optimization_layer, num_iterations, num_repetitions, blend)

		image_path = results["image_path"]

		# constructing email
		msg = Message("Deep Dream Results",
		              sender="RocketAIcs01430project@gmail.com",
		              recipients=[email_address])

		with app.open_resource(image_path) as fp:
			msg.attach("processed_image.png", "image/png", fp.read())

		# sending email
		mail.send(msg)
		print("email delivered")


@app.route("/")
def main():
    return render_template("index.html")

@app.route("/request-image", methods=["POST"])
def requestImage():
	print("received POST request")
	print(request.form)
	print(request.files)

	# getting data from the post request
	image_file = request.files['submit-image']
	image_bytes = image_file.read()
	image = np.float32(Image.open(io.BytesIO(image_bytes)))
	email = request.form.get('email', 'example@example.com')

	optimization_layer = int(request.form.get('optimization-layer', '6'))
	num_iterations = int(request.form.get('num-iterations', '3'))
	num_repetitions = int(request.form.get('num-repetitions', '3'))
	blend = float(request.form.get('blend', '0.2'))

	print("request details: ", email, optimization_layer, num_iterations, num_repetitions, blend)
	# initializing the image processing
	processing_thread = threading.Thread(target=initiate_image_processing, args=[image, email, optimization_layer, num_iterations, num_repetitions, blend])
	processing_thread.start()

	response = {"status": 200}
	return

if __name__ == "__main__":
    app.run(debug=True)