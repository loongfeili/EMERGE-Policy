"""One FIFO quota coordinator for all evaluation processes; never receives LLM keys."""
import argparse
from collections import deque
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import socket
import threading
import time


class Pacer:
    def __init__(self, rpm, cooldown=65.0):
        if rpm <= 0:
            raise ValueError('rpm must be positive')
        self.rpm = float(rpm)
        self.cooldown = cooldown
        self.condition = threading.Condition()
        self.queue = deque()
        self.next_at = 0.0
        self.last_limited = float('-inf')
        self.grants = 0
        self.rate_limit_events = 0

    def acquire(self):
        ticket = object()
        started = time.monotonic()
        with self.condition:
            self.queue.append(ticket)
            try:
                while True:
                    now = time.monotonic()
                    if self.queue[0] is ticket and now >= self.next_at:
                        self.next_at = now + 60.0 / self.rpm
                        self.grants += 1
                        return {'granted': True, 'wait_seconds': now-started, 'rpm': self.rpm}
                    self.condition.wait(timeout=max(.001, min(1.0, self.next_at-now)) if self.queue[0] is ticket else 1.0)
            finally:
                self.queue.remove(ticket)
                self.condition.notify_all()

    def limited(self):
        with self.condition:
            now = time.monotonic()
            self.rate_limit_events += 1
            if now-self.last_limited >= self.cooldown:
                self.rpm = max(1.0, self.rpm*.75)
                self.last_limited = now
            self.next_at = max(self.next_at, now+self.cooldown)
            self.condition.notify_all()

    def status(self):
        with self.condition:
            return {'status':'healthy', 'rpm':self.rpm, 'waiting':len(self.queue),
                'grants':self.grants, 'rate_limit_events':self.rate_limit_events,
                'next_grant_seconds':max(0,self.next_at-time.monotonic())}


def server(host, port, rpm, token):
    if not token:
        raise ValueError('EMERGE_RATE_LIMIT_TOKEN is required')
    pacer = Pacer(rpm)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, value):
            body = json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # An abandoned permit stays consumed, preserving the rate bound.

        def do_GET(self):
            self.reply(200 if self.path=='/healthz' else 404,
                pacer.status() if self.path=='/healthz' else {'error':'not_found'})

        def do_POST(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''), 'Bearer '+token):
                self.reply(403,{'error':'forbidden'})
                return
            size = int(self.headers.get('Content-Length','0'))
            if not 0 <= size <= 1024:
                self.reply(413,{'error':'invalid_payload_size'})
                return
            self.rfile.read(size)
            if self.path == '/acquire':
                self.reply(200,pacer.acquire())
            elif self.path == '/rate-limited':
                pacer.limited()
                self.reply(200,pacer.status())
            else:
                self.reply(404,{'error':'not_found'})

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ':' in host else socket.AF_INET
        request_queue_size = 256
        daemon_threads = True

    return Server((host,port),Handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='::')
    parser.add_argument('--port',type=int,default=8100)
    parser.add_argument('--rpm',type=float,required=True)
    args=parser.parse_args()
    server(args.host,args.port,args.rpm,os.environ['EMERGE_RATE_LIMIT_TOKEN']).serve_forever()
