from flask import Flask, request, jsonify, send_file
from io import BytesIO
from generator import generate_image
import os
import time

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_image_endpoint():
    """
    Expects a JSON payload with:
    {
      "prompt": "Text prompt for image generation",
      "height": (optional) Image height,
      "width": (optional) Image width,
      "model": (optional) Model to use for generation
    }
    Returns the generated image as a PNG file.
    """
    try:
        start_time = time.time()
        print("Processing generate_image request")
        
        data = request.get_json()
        prompt = data.get('prompt')
        height = data.get('height', 1024)
        width = data.get('width', 1024)
        model = data.get('model', 'flux')
        
        if not prompt:
            return jsonify({"error": "'prompt' is required."}), 400
        
        # Generate image
        generation_start = time.time()
        image = generate_image(prompt, height, width, model)
        generation_time = time.time() - generation_start
        
        # Convert image to bytes for response
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        
        print(f"Image generation completed in {generation_time:.2f} seconds")
        print(f"Total request processing time: {time.time() - start_time:.2f} seconds")
        
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        print(f"Error in generate_image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "ok",
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/system-info', methods=['GET'])
def system_info():
    """Endpoint to check system configuration including GPU status"""
    try:
        gpu_info = {}
        
        # Try to get PyTorch GPU info
        try:
            import torch
            gpu_info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                gpu_info["cuda_device_count"] = torch.cuda.device_count()
                gpu_info["cuda_device_name"] = torch.cuda.get_device_name(0)
                gpu_info["cuda_version"] = torch.version.cuda
        except ImportError:
            gpu_info["torch_import_error"] = "PyTorch not installed"
        
        return jsonify({
            "status": "ok",
            "gpu_info": gpu_info,
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"System info check failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Simple landing page with API documentation"""
    return jsonify({
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
                    "model": "(optional) Model to use (default: 'flux')"
                }
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint"
            }
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8484, debug=False, threaded=False)