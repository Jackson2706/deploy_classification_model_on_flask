import onnxruntime
import cv2
import numpy as np
from typing import Tuple, List


def softmax(logits):
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits)
    return probabilities


class ClassificationInferenceModel:
    def __init__(self, cls_onnx_model_path: str, input_shape: Tuple, label_list: List):
        self.session = onnxruntime.InferenceSession(cls_onnx_model_path)
        self.input_shape = input_shape
        self.label_list = label_list

    def _preprocessing(self, frame):
        input_img = cv2.resize(frame, self.input_shape)
        input_img = input_img.astype(np.float32)
        # Normalize image
        input_img /= 255.0  # Scale pixel values to [0, 1]
        mean = np.array([0.5, 0.5, 0.5])  # Mean for normalization
        std = np.array([0.5, 0.5, 0.5])  # Standard deviation for normalization
        input_img -= mean  # Subtract mean
        input_img /= std # Divide by standard deviation
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :]
        return input_tensor

    def _execute(self, input_data):
        return self.session.run(
            None,
            {self.session.get_inputs()[0].name: input_data}
        )

    def _postprocessing(self, output_data):
        output_data = np.array(output_data)  # Convert to numpy array if not already
        probabilities = softmax(output_data)
        max_index = np.argmax(probabilities)
        return self.label_list[max_index]

    def run_native_inference(self, frame):
        input_data = self._preprocessing(frame)
        output_data = self._execute(input_data)
        results = self._postprocessing(output_data)
        return results


if __name__ == "__main__":
    image = cv2.imread("../test_2.PNG")
    model = ClassificationInferenceModel(cls_onnx_model_path="../weights/cls_model.onnx", input_shape=(32, 32), label_list =[str(i) for i in range(10)])
    result = model.run_native_inference(image)
    print(result)
