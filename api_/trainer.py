from http.server import BaseHTTPRequestHandler


# Modify your logic function to accept parameters
def run_training_logic(learning_rate, num_epochs):
    """
    This is where your actual training code would use the parameters.
    """
    print(f"Starting training with LR: {learning_rate} and Epochs: {num_epochs}")
    # ... your actual training code using torch, onnx, etc. ...

    # Return a message confirming the parameters were received
    return {
        "status": "success",
        "message": f"Training completed with LR={learning_rate}, Epochs={num_epochs}.",
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # 1. Get the content length from the request headers
        try:
            content_length = int(self.headers["Content-Length"])
        except TypeError:
            content_length = 0

        # 2. Read the raw request body
        post_data_bytes = self.rfile.read(content_length)

        # 3. Decode from bytes to string
        params = post_data_bytes.decode("utf-8")

        # 4. Compute the return
        result = str(int(params) * 3)

        # Send the response back to the frontend
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        _ = self.wfile.write(result.encode("utf-8"))
