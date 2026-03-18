from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 7071


class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(f"receive GET:{self.path}")

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h1>Hello world</h1>")


server = HTTPServer(("localhost", PORT), MyHandler)
print(f"Server running at http://localhost:{PORT}")
server.serve_forever()
