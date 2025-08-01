
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class SimpleWebServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = f"<!DOCTYPE html><html><head><title>Simple Web Server</title></head><body><h1>Welcome to Simple Web Server</h1><p>This is a basic web server running on Python.</p><p>Current path: {self.path}</p></body></html>"
        
        self.wfile.write(html.encode())
        
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            'status': 'success',
            'message': 'Data received',
            'data': post_data.decode()
        }
        
        self.wfile.write(json.dumps(response).encode())

def main():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, SimpleWebServer)
    print("Server running on http://localhost:8000")
    httpd.serve_forever()

if __name__ == "__main__":
    main()
