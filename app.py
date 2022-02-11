from crypt import methods
from unittest import result
import flask
import numpy as np
import pickle

model = pickle.load(open('model/classifier_model.pkl','rb'))

app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    return(flask.render_template('main.html'))

@app.route('/predict',methods=['POST'])
def process():
    features = [float(x) for x in flask.request.form.values()]
    final_features = [np.array(features)]
    result = model.predict(final_features)

    output = {0:'tipe nasabah yang memiliki balance moderat, sangat jarang melakukan transaksi pembelian, lebih sering melakukan transaksi dengan uang tunai dimuka, hampir tidak pernah melakukan pembelian dengan metode mencicil. tipe user ini memiliki limit kartu kredit medium', 
    1:'Memiliki balance dan limit kartu kredit paling tinggi, lebih sering melakukan pembelian dengan metode sekali bayar(one off purchases), sering melakukan transaksi belanja, hampir tidak pernah melakukan pembelian dengan uang tunai dimuka ',
    2:'Memiliki balance paling rendah diantara tipe nasabah lain, frekuensi pembelian cukup tinggi dan sering melakukan pembelian dengan metode pembayaran mencicil, memiliki limit kartu kredit paling rendah'}
    return flask.render_template('main.html',result_text='{}'.format(output[result[0]]))

if __name__=='__main__':
    app.run(debug=True)