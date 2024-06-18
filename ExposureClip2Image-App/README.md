# 1. Clone or Download the Repository

- Download the source code to your local machine, or clone the repository.

```` bash
git clone https://github.com/mi-classroom/mi-web-technologien-beiboot-ss2024-MaikRoth.git
cd mi-web-technologien-beiboot-ss2024-MaikRoth
````

# 2. Navigate to the Project Directory

- Open a terminal (or command prompt) and navigate to the directory where the code is located.

```` bash
cd ExposureClip2Image-App
````


# 3. Create and Activate a Virtual Environment (Optional but Recommended)

- Creating a virtual environment helps manage dependencies and avoid conflicts.

```` bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
````

# 4. Install Required Dependencies

- Use pip to install the necessary Python libraries.
  
```` bash
pip install Flask werkzeug opencv-python numpy
````

# 5. Create Necessary Directories
 
- Ensure the required directories for uploads, frames, and outputs exist.

```` bash
# Create directories if they don't exist
mkdir -p static/uploads static/frames outputs
````

# 6. Run the Flask Application

- Execute the Flask application by running the following command:

```` bash
python app.py
````

# 7. Access the Application in a Web Browser

- Once the server is running, open your web browser and navigate to:

```` bash
http://127.0.0.1:5000
````