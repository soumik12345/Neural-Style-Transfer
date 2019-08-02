import warnings
warnings.filterwarnings('ignore')
from StyleTransferer import StyleTransferer

transferer = StyleTransferer('./Artworks/422652.jpg', './TestCases/5726.jpg', num_iterations = 1)