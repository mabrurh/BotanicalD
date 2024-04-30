from flask import Flask, request, render_template, jsonify
import sqlite3

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('Plants_For_a_Future_Updated.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/search', methods=['POST'])
def search():
    search_query = request.form['species_name'].strip()  # .strip() to remove any extra whitespace
    conn = get_db_connection()
    # Adjust the SQL query to use LIKE for flexible matching and handle case insensitivity
    plant = conn.execute('SELECT * FROM "Species List" WHERE LatinName LIKE ? OR CommonName LIKE ?',
                         ('%' + search_query + '%', '%' + search_query + '%')).fetchone()
    conn.close()
    if plant:
        return render_template('results1.html', plant=plant)
    else:
        return render_template('index1.html', error='No details found for the entered species.')

if __name__ == '__main__':
    app.run(debug=True)
