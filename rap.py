from flask import Flask, render_template, request
import finalop as fo

app = Flask(__name__)

# home page route
@app.route('/')
def home():
    return render_template('home.html')

# upload image route
@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['Image']
        if file :
            file_type=fo.get_input(file)
            return render_template("home.html",string="the given file is a:",file_type=file_type)
        else:
            return render_template("home.html",string="please enter a file",file_type="")

if __name__ == '__main__':
    app.run(debug=True)
