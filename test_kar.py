from flask import Flask
from flask import Response
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/test_kar', methods=['POST', 'GET'])
@cross_origin()
def test_kar():
    with open("time.csv") as fp:
        cs = fp.read()
    return Response(
        cs,
        mimetype="text/csv",
        headers={"Content-disposition":
                     "attachment; filename=time_res.csv"})


if __name__ == "__main__":
    app.run(host="localhost", port=8080)


