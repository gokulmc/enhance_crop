import torch
from ...utils.Util import log
class UpscaleModelWrapper:
    def __init__(self, model: torch.nn.Module, device: torch.device, precision: torch.dtype, backed:str="pytorch"):
        self.__model = model
        self.__device = device
        self.__precision = precision
        model.eval().to(device, dtype=precision)
    
    def setPrecision(self, precision: torch.dtype):
        self.__precision = precision
        self.__model.to(self.__device, dtype=precision)
    
    @torch.inference_mode()
    def loadModel(
        self, modelPath: str, dtype: torch.dtype = torch.float32, device: str = "cuda"
    ) -> torch.nn.Module:
        try:
            from . import ModelLoader, ImageModelDescriptor, UnsupportedModelError
        except ImportError:
            # spandrel will import like this if its a submodule
            from .libs.spandrel.spandrel import ModelLoader, ImageModelDescriptor, UnsupportedModelError
        try:
            model = ModelLoader().load_from_file(modelPath)
            assert isinstance(model, ImageModelDescriptor)
            self.model = model
            # get model attributes
            
        except (UnsupportedModelError) as e:
            log(f"Model at {modelPath} is not supported: {e}")
            raise e

        self.scale = model.scale
        model = UpscaleModelWrapper(model=model, device=device, precision=dtype)
        
        try:
            example_input = torch.zeros((1, 3, 64, 64), device=self.device, dtype=self.dtype)
            model(example_input)
        except Exception as e:
            print("Error occured during model validation, falling back to float32 dtype.\n")
            log(str(e))
            model.setPrecision(torch.float32)
            
        return model

    def getModel(self):
        return self.__model

    def predict(self, input_data):
        # Preprocess input data
        processed_data = self.preprocess(input_data)
        
        # Make prediction using the wrapped model
        prediction = self.__model(processed_data)
        
        # Postprocess the prediction
        result = self.postprocess(prediction)
        
        return result

    def preprocess(self, input_data):
        # Implement preprocessing logic here
        return input_data

    def postprocess(self, prediction):
        # Implement postprocessing logic here
        return prediction