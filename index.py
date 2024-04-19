# index.py
from app import app, server  # Import the 'app' and 'server' instances from your main app module

# This ensures that the `index.py` serves as the entry point for your application.
# Google App Engine requires this for routing and managing the application.

if __name__ == '__main__':
    # App Engine instances have a pre-configured port set via the environment variable 'PORT'
    # There's generally no need to specify it manually unless for local testing
    app.run_server(debug=False)
