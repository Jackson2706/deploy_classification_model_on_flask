import unittest
import os
from app import app


class FlaskTest(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.app = app.test_client()

    def tearDown(self):
        pass

    def test_index(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Image Uploader", response.data)

    def test_upload(self):
        with open("test_2.PNG", "rb") as f:
            data = {"file": (f, "test_2.PNG")}
            response = self.app.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Uploaded Image", response.data)

    def test_predict(self):
        with open("test_2.PNG", "rb") as f:
            data = {"file": (f, "test_2.PNG")}
            self.app.post('/upload', data=data, content_type='multipart/form-data')
        response = self.app.post('/predict', data={"filename": "test_2.PNG"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Result", response.data)


if __name__ == '__main__':
    unittest.main()
