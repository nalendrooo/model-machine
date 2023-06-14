import os
# import uvicorn
import traceback
import numpy as np
import tensorflow_text
import tensorflow as tf
from gunicorn.app.base import BaseApplication
# import download

from pydantic import BaseModel
from urllib.request import Request
from fastapi import FastAPI, Response


# URL to download the protobuf model
url = "https://storage.googleapis.com/captone-bucket-model123/saved_model/1/saved_model.pb"

# File path to save the downloaded model
model_path = "./saved_model.pb"

# Download the model
urllib.request.urlretrieve(url, model_path)

# Initialize Model
model = tf.saved_model.load("./saved_model.pb")
# model = tf.saved_model.load("./saved_model/1")

app = FastAPI()

# This endpoint is for a test to this server


@app.get("/")
def index():
    return "Hello world from ML endpoint!"

# If your model needs text input, use this endpoint!


class RequestText(BaseModel):
    text: str


@app.post("/predict_text")
def predict_text(req: RequestText, response: Response):
    try:
        # In here you will get text sent by the user
        text = req.text
        print("Uploaded text:", text)

        # Step 1: Text preprocessing
        def preprocess_text(text):
            processed_text = text.lower()
            return processed_text

        # Step 2: Prepare your data for the model
        def prepare_data(input_data):
            prepared_data = preprocess_text(input_data)
            return [prepared_data]

        # Step 3: Predict the data
        def predict_data(data):
            result = model(tf.constant(data))
            return result

        # Step 4: Change the result to your determined API output
        def format_output(result):
            # Modify this function according to your model's output format
            labels = ['Teknik Informatika, Sistem Informasi, Ilmu Komputer',
                      'Ekonomi, Akuntansi, Manajemen',
                      'Seni, Desain Komunikasi Visual, Desain Produk',
                      'Kedokteran, Kesehatan Masyarakat, Keperawatan']
            predicted_index = np.argmax(result)
            output = {"predicted_jurusan": labels[predicted_index]}
            return output

        # Preprocess the text
        preprocessed_text = preprocess_text(text)

        # Prepare the data
        data = prepare_data(preprocessed_text)

        # Predict the data
        prediction = predict_data(data)

        # Format the output
        output = format_output(prediction)

        return {"prediction": output}

    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"


# Starting the server
# You can check the API documentation easily using /docs after the server is running
# port = os.environ.get("PORT", 8080)
# print(f"Listening to http://0.0.0.0:8080")
# uvicorn.run(app, host='0.0.0.0', port=8080)
# download.run()

class Server(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        for key, value in self.options.items():
            self.cfg.set(key, value)

    def load(self):
        return self.application


if __name__ == '__main__':
    # app.run(debug=True)
    options = {
        'bind': '0.0.0.0:5000',
        'workers': 4
    }
    server = Server(app, options)
    server.run()
