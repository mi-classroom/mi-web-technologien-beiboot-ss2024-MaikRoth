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

# Using Docker Compose

For an easier setup, you can use Docker Compose. Follow the steps [here](#setup-using-docker-compose) to get started. First do Step 1 and Step 2 and then go on from here.


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
pip install -r requirements.txt
````

# 5. Run the Flask Application

- Execute the Flask application by running the following command:

```` bash
python app.py
````

# 6. Access the Application in a Web Browser

- Once the server is running, open your web browser and navigate to:

```` bash
http://127.0.0.1:5000
````
# Setup using Docker Compose
# 3. Ensure Docker and Docker Compose are Installed

- Make sure you have Docker and Docker Compose installed on your machine. You can download them from Docker's official website.

# 4. Build and Run the Application using Docker Compose
Use the following commands to build the Docker image and start the application:

````bash
docker-compose up -d
````

# 5. Access the Application in a Web Browser
- Once the containers are running, open your web browser and navigate to:
````bash
http://localhost:5000
````
