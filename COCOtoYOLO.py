import pylabel
from pylabel import importer
dataset = importer.ImportVOC(path='C:/Users/jorge/Desktop/RDD-YOLOv8/datasets/train/train')
dataset.export.ExportToYoloV5()