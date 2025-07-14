class HyperParameters:
    def __init__(self):
        self.max_iter = 200
        self.dc = 64
        self.lr = 0.0001
        self.Lambda = 0.001
        self.batch_size = 64
        self.model_path = 'model'
        self.test_data_path = 'data/test_img'
        self.test_data_result_path = 'picture/test_img_result'
        self.result_path = 'picture'
        self.cut_range = (48, 80)

    def __str__(self):
        return (
            f"Iterations: {self.max_iter}\n"
            f"Depth Channel (dc): {self.dc}\n"
            f"Learning Rate: {self.lr}\n"
            f"Lambda: {self.Lambda}\n"
            f"Batch_size: {self.batch_size}\n"
            f"Model Path: {self.model_path}\n"
            f"Test Data Path: {self.test_data_path}\n"
            f"Test Data Result Path: {self.test_data_result_path}\n"
            f"Result Path: {self.result_path}\n"
            f"Cut Range: {self.cut_range}\n"
        )
