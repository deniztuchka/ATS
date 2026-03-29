import os
from flask import Flask, send_from_directory
from app.api.routes import api_bp
from app.services.logging_conf import configure_logging
from app.config import AppConfig

BASE_DIR = os.path.dirname(__file__)              # .../thesis/app
WEB_DIR  = os.path.join(BASE_DIR, "../web")          # .../thesis/app/web
ASSETS_DIR = os.path.join(WEB_DIR, "assets")      # .../thesis/app/web/assets

def create_app():
    app = Flask(__name__, static_folder=None)
    app.config.from_object(AppConfig)
    configure_logging(app)

    app.register_blueprint(api_bp, url_prefix="/api")

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/assets/<path:path>")
    def assets(path):
        return send_from_directory(ASSETS_DIR, path)

    return app

if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
