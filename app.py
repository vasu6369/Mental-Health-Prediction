from flask import Flask, render_template
from api import api

def create_app():
    app = Flask(__name__, template_folder="templates")

    app.register_blueprint(api)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app

app = create_app()
if __name__ == "__main__":
    
    app.run(debug=True)
