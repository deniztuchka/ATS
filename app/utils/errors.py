class HTTPError(Exception):
    def __init__(self, status=400, message="Bad Request"):
        super().__init__(message)
        self.status = status
        self.message = message
