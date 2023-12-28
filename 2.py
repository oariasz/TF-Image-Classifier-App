# Basic ML App
# Images Classifier (Animals and Vehicles)
# Already on Github
# Branch: Universo Paralelo



from taipy.gui import Gui
from tensorflow.keras import models
from PIL import Image
import numpy as np





# Class names to predict
class_names = {
    0: 'Avión',
    1: 'Automóvil',
    2: 'Ave',
    3: 'Gato',
    4: 'Venado',
    5: 'Perro',
    6: 'Rana',
    7: 'Caballo',
    8: 'Barco',
    9: 'Camión',
}

model = models.load_model("nn_classifier_model.keras")


def predict_image(model, path_to_img):
    img = Image.open(path_to_img)
    img = img.convert("RGB")
    img = img.resize((32, 32))
    data = np.asarray(img)
    data = data / 255
    probs = model.predict(np.array([data])[:1])

    top_prob = probs.max()
    top_pred = class_names[np.argmax(probs)]
    
    return top_prob, top_pred


content = ''
img_path = 'placeholder_image.png'
prob = 0
pred = ''

# Layout de la página index
index = """
<|text-center|
<|{"Logo Estratek blanco.png"}|image|width=25vw|>

<|{content}|file_selector|extensions=.png|>
Selecciona una image a clasificar

<|{pred}|font-size=36pt|>

<|{img_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>

>
"""
# <|{certainty_text}|text|"Certeza:">

def on_change(state, var_name, var_val):
    if var_name == 'content':
        top_prob, top_pred = predict_image(model, var_val)
        state.prob = round(top_prob * 100)
        state.pred = "Esto está clasificado como: " + top_pred
        state.img_path = var_val
    #print(var_name, var_val)
    
    

app = Gui(page=index)

if __name__ == "__main__":
    app.run(port=8002, use_realoader=True)




