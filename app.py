from flask import Flask, render_template, request
from keras import models
from backtester import model_check
import io
import sys
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    plot_url = None
    if request.method == 'POST':
        model_path = request.form['model_path']
        num_test_images = int(request.form.get('num_test_images', 1000))
        plot = int(request.form.get('plot', 1))
        
        try:
            user_model = models.load_model(model_path)
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            model_check(model=user_model, plot=plot, num_test_images=num_test_images)
            
            sys.stdout = old_stdout
            output = buffer.getvalue()
            
            lines = output.strip().split('\n')
            result = {}
            for line in lines:
                if "Number of correctly predicted samples:" in line:
                    result['correct_predictions'] = int(line.split(': ')[1])
                elif "Number of incorrectly predicted samples:" in line:
                    result['incorrect_predictions'] = int(line.split(': ')[1])
                elif "Accuracy of the model:" in line:
                    result['accuracy'] = float(line.split(': ')[1].rstrip('%'))
            
            if plot == 1:
                img = io.BytesIO()
                plt.savefig(img, format='png', facecolor='#1e1e1e', edgecolor='none')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                plt.close()
        
        except Exception as e:
            result = {"error": str(e)}
    
    return render_template('index.html', result=result, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)