import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = "huan.h5"
SAMPLES_TO_CONSIDER = 22050

class _KeyWord_Spotting_Service:

    model = None

    _mappings = [
        "eight",
        "five",
        "four",
        "nine",
        "one",
        "seven",
        "six",
        "three",
        "two",
        "zero"
    ]

    _instance = None

    def predict(self, file_path):
        MFCCs = self.preprocess(file_path)

        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        # make_data.prepare_data(file_path)
        # temp_file = os.path.join(os.path.curdir, os.path.basename(file_path).split('.')[0]) + '.wav'
        signal, sr = librosa.load(file_path)

        if (len(signal) >= SAMPLES_TO_CONSIDER):
            signal = signal[:SAMPLES_TO_CONSIDER]
        else:
            signal = librosa.util.fix_length(signal, SAMPLES_TO_CONSIDER)
        
        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return MFCCs.T

def Keyword_Spotting_Service():
    if _KeyWord_Spotting_Service._instance is None:
        _KeyWord_Spotting_Service._instance = _KeyWord_Spotting_Service()
        _KeyWord_Spotting_Service.model = keras.models.load_model(MODEL_PATH)

    return _KeyWord_Spotting_Service._instance

if __name__ == "__main__":

    kss = Keyword_Spotting_Service()
    keyword1 = kss.predict('test/test0.wav')
    keyword2 = kss.predict('test/test1.wav')
    keyword3 = kss.predict('test/s7ven.wav')
    keyword4 = kss.predict('test/test3.wav')
    keyword5 = kss.predict('test/two2.wav')
    

    print(f"Predicted keyword: {keyword1}, {keyword2}, {keyword3}, {keyword4}, {keyword5}")