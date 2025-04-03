from flask import Flask, request, jsonify, send_file
from io import BytesIO
import time
from generator import ImageGenerator
import threading
from queue import Queue
import concurrent.futures

app = Flask(__name__)
request_queue = Queue()
thread_lock = threading.Lock()

try:
    generator = ImageGenerator()
except RuntimeError as e:
    print(f"Failed to initialize ImageGenerator: {e}")
    generator = None

# Create a thread pool executor
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

def make_response(status="ok", data=None, error=None, http_code=200):
    response = {
        "status": status,
        "timestamp": time.time()
    }
    if data is not None:
        response["data"] = data
    if error is not None:
        response["error"] = error
    return jsonify(response), http_code

def generate_image_task(prompt, height, width, model):
    with thread_lock:
        return generator.generate(prompt, height, width, model)

@app.route('/generate', methods=['POST'])
def generate_image_endpoint():
    if generator is None:
        return make_response(
            status="error",
            error="GPU with CUDA support is required but not available",
            http_code=503
        )

    try:
        start_time = time.time()
        print("Processing generate_image request")
        
        data = request.get_json()
        prompt = data.get('prompt')
        height = data.get('height', 1024)
        width = data.get('width', 1024)
        model = data.get('model', 'flux-schnell')
        
        if not prompt:
            return make_response(
                status="error",
                error="'prompt' is required",
                http_code=400
            )
        
        generation_start = time.time()
        
        # Submit the task to the thread pool
        future = executor.submit(generate_image_task, prompt, height, width, model)
        image = future.result()  # Wait for the result
        
        generation_time = time.time() - generation_start
        
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        
        print(f"Image generation completed in {generation_time:.2f} seconds")
        print(f"Total request processing time: {time.time() - start_time:.2f} seconds")
        
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        print(f"Error in generate_image: {e}")
        return make_response(
            status="error",
            error=str(e),
            http_code=500
        )

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        return make_response(status="ok")
    except Exception as e:
        print(f"Health check failed: {e}")
        return make_response(
            status="error",
            error=str(e),
            http_code=500
        )

@app.route('/system-info', methods=['GET'])
def system_info():
    """Endpoint to check system configuration including GPU status"""
    if generator is None:
        return make_response(
            status="error",
            error="GPU with CUDA support is required but not available",
            http_code=503
        )

    try:
        return make_response(
            status="ok",
            data={"system_info": generator.get_system_info()}
        )
    except Exception as e:
        print(f"System info check failed: {e}")
        return make_response(
            status="error",
            error=str(e),
            http_code=500
        )

@app.route('/', methods=['GET'])
def index():
    """Simple landing page with API documentation"""
    api_docs = {
        "name": "Flux Image Generation API",
        "endpoints": {
            "/generate": {
                "method": "POST",
                "content_type": "application/json",
                "description": "Generate image from text prompt",
                "parameters": {
                    "prompt": "Text prompt for image generation",
                    "height": "(optional) Image height (default: 1024)",
                    "width": "(optional) Image width (default: 1024)",
                    "model": "(optional) Model to use (default: 'flux-schnell')"
                }
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint"
            },
            "/system-info": {
                "method": "GET",
                "description": "System configuration and GPU status"
            }
        }
    }
    return make_response(status="ok", data=api_docs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8484, debug=False, threaded=True)
