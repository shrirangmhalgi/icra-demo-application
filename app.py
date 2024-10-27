



from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import os
from result_generator import generate_results
from datetime import datetime

app = Flask(__name__)

# Hardcoding a simple username and password for demonstration
valid_username = "admin"
valid_password = "admin"

# Set the upload folder and allowed extensions
RGB_FOLDER = 'static/uploads/rgb'
SPARSE_DEPTH_FOLDER = 'static/uploads/sparse-depth'
RESULTS_FOLDER = 'static/uploads/results'

# configure the image folders
app.config['RGB_FOLDER'] = RGB_FOLDER
app.config['SPARSE_DEPTH_FOLDER'] = SPARSE_DEPTH_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Limit file size to 5 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# define the allowed extensions
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def handle_login():
    username = request.form['username']
    password = request.form['password']
    
    print(username, password)

    # Check if the provided username and password match
    if username == valid_username and password == valid_password:
        # On successful login, redirect to the second page
        return redirect(url_for('dashboard'))
    else:
        # Reload login page on failure (you can add flash messages here)
        return render_template('login.html', error="Invalid credentials")

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# Route to handle the file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the response has both files
    if 'rgb_file' not in request.files or 'sparse_depth_file' not in request.files:
        return jsonify(success=False)

    # read the rgb and sparse depth files
    rgb_file = request.files['rgb_file']
    sparse_depth_file = request.files['sparse_depth_file']

    # get the selected model
    model = request.form.get('selected_option')

    if rgb_file.filename == '' or sparse_depth_file.filename == '':
        return jsonify(success=False)

    # calculate the size of the files
    rgb_file.seek(0, 2)  
    rgb_file_size = rgb_file.tell()  
    rgb_file.seek(0)  
    
    sparse_depth_file.seek(0, 2)
    sparse_depth_file_size = sparse_depth_file.tell()
    sparse_depth_file.seek(0)

    # check if the size is valid
    if sparse_depth_file_size > app.config['MAX_CONTENT_LENGTH'] or rgb_file_size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify(success=False, message="File size too large")

    if sparse_depth_file and allowed_file(sparse_depth_file.filename) and rgb_file and allowed_file(rgb_file.filename):
        # rename the rgb and sparse files uniquely according to timestamp
        current_timestamp = datetime.now()

        rgb_filename = f"rgb_{current_timestamp}.{rgb_file.filename.split('.')[-1]}"
        rgb_file.save(os.path.join(app.config['RGB_FOLDER'], rgb_filename))

        sparse_depth_filename = f"sparse_depth_{current_timestamp}.{sparse_depth_file.filename.split('.')[-1]}"
        sparse_depth_file.save(os.path.join(app.config['SPARSE_DEPTH_FOLDER'], sparse_depth_filename))

        # call the ML code to generate the results...... save the files in the correct format
        dense_depth_filename = generate_results(model, current_timestamp)

        # if something fails
        if dense_depth_filename is None:
            return jsonify(success=False)
        
        # return the results 
        return jsonify(success=True, filename=dense_depth_filename)
    return jsonify(success=False)

if __name__ == '__main__':
    os.makedirs('static/uploads/', exist_ok=True)
    os.makedirs(app.config['RGB_FOLDER'], exist_ok=True)
    os.makedirs(app.config['SPARSE_DEPTH_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    app.run(debug=True)
