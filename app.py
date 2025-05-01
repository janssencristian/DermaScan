import cv2
import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import efficientnet.tfkeras as efn

# Carregando o modelo treinadomodel = tf.keras.models.load_model(
model = tf.keras.models.load_model(
    "dermascan_model.keras",
    custom_objects={"EfficientNetB3": efn.EfficientNetB3}
)

# Classes da base
class_names = ['bkl', 'mel', 'nv',]

# Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# Função principal
def predict_with_explainability(pil_img):
    img = pil_img.resize((300, 300))
    img_array = img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array_exp)
    pred_idx = np.argmax(preds)
    class_name = class_names[pred_idx]
    confidence = float(np.max(preds))

    # Grad-CAM
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    heatmap = make_gradcam_heatmap(img_array_exp, model, last_conv_layer)
    heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(img), 0.6, heatmap_colored, 0.4, 0)
    result_img = Image.fromarray(superimposed_img)

    return result_img, f"Classe prevista: {class_name} (Confiança: {confidence:.2%})"

# Interface Gradio
interface = gr.Interface(
    fn=predict_with_explainability,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(label="Explicabilidade Grad-CAM"), gr.Text(label="Resultado")],
    title="DermaScan - Diagnóstico de Lesões de Pele",
    description="Carregue uma imagem de lesão dermatológica. O modelo exibirá a classe prevista e destacará visualmente as regiões mais importantes para a decisão."
)

interface.launch()

#ceratose benigna (BKL), nevo melanocítico (NV), melanoma (MEL).