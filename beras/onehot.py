import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """
    def __init__(self):
        super().__init__()
        self.fitted_ = False
        self.label2idx_ = {}
        self.idx2label_ = {}
        self.num_classes_ = 0

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        unique_labels = np.unique(data)
        self.num_classes_ = len(unique_labels)
        
        # 建立 label -> index 和 index -> label 的字典
        self.label2idx_ = {label: i for i, label in enumerate(unique_labels)}
        self.idx2label_ = {i: label for i, label in enumerate(unique_labels)}
        
        self.fitted_ = True

    def forward(self, data):
        #return NotImplementedError
        if not self.fitted_:
            # 如果尚未 fit，则先使用当前 data 做 fit
            self.fit(data)

        # 准备输出矩阵 (batch_size, num_classes_)
        batch_size = len(data)
        encoded = np.zeros((batch_size, self.num_classes_), dtype=np.float32)

        # 每个 label 根据其在字典中的 index，将相应位置设为 1
        for i, label in enumerate(data):
            idx = self.label2idx_[label]  # 找到对应的index
            encoded[i, idx] = 1.0

        return encoded

    def inverse(self, data):
        #return NotImplementedError
        batch_size = data.shape[0]
        decoded = np.zeros((batch_size,), dtype=np.int32)  # 或者根据你的 label 类型设置

        for i in range(batch_size):
            # 找到该行向量的最大值位置 => argmax => index => label
            idx = np.argmax(data[i])
            decoded[i] = self.idx2label_[idx]

        return decoded
