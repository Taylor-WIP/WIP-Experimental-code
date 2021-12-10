import tensorflow as tf
from models import *
from train import Train
from generate_dataset import GenerateSalDataset
from tests import *

from datasets import *



###### TRAIN PRIMARY ######
## trains a model, in this case MiniResNetK, which I added as an example (line 322 in models.py)
#### on cifar10 (line 98 in datasets.py). Saves as modelname_datasetname.h5 - so in this case as MiniResNetK_cifar10.h5
#
# Train(
#     MiniResNetK, epochs=20, dataset=Cifar10, batch_size=32
# ).train()
#

###### GENERATE DATASET FROM TRAINED PRIMARY ######
### Once the model is trained (there isn't currently a trained model callded MiniResNetK_cifar10.h5 as I've added it as and example)
### we can generate a dataset from it - this basically runs Cifar10 through the trained model to get maps for each of the 3 saliency methods
## At the momement you have to change the save pathe manually in generate_dataset.py (I want to change this at some point to make it easier to use)
## Currently it is set to save in the folder with 8 other MiniResNets.

### I've left it commented out as it takes a long time to run and will start saving files imediatly (which are annoying to delete)so you don't want to run it by mistake


##### NEW: experiment_directory #####
# previously had no concret convention for the save path for datasets,
# will going forward try to save as "PrimaryNetworkType/ExperimentName".
# The default for experiment_directory is "other", so if we forget to set the directory it at least doesn't go to the wrong folder or save over anything.

#### ALSO NEW: #######
# Generating datasets will now save maps as "input_id_label" - so we have the original input id saved with the maps.
#  Note - currently this only works for CIFAR10. Set get_id to True in order to save in this format (defualt is False)

# GenerateSalDataset(
#     model=MiniResNetK,
#     model_path="../models/MiniResNetK_cifar10.h5",
#     dataset=Cifar10,
#     experiment_directory="ResNets/ManyResNets/EleventhResNet",
#     # get_id = True
# ).generate_dataset()


####### TRAIN SECONDARY ###### (note, this is the same as training a primary, just with a different dataset)
## as an example, you can run the below, but it will save over an existing secondary. This one is for the recent experiment calssifying the maps by input class
## This one is for model "A" for Grad


# Train(
#     MiniCNN, epochs=20, dataset=ResAByClassCifar10Grad, batch_size=32
# ).train()

######## EVALUATE MODEL #########
# this will evaluate the model trained above with the "A" dataset on the "B" network's dataset

# EvaluateModel(
#     MiniCNN,
#     "../models/MiniCNN_ResA_ByClassCifar10_Grad.h5",
#     ResBByClassCifar10Grad,
# ).evaluate()
#









########### RECENT ############

#
# Train(
#     MiniCNN, epochs=20, dataset=LayersCNNs2c6Grad, batch_size=32
# ).train()
#
# Train(
#     MiniCNN, epochs=20, dataset=LayersCNNs2c6SG, batch_size=32
# ).train()
#
# Train(
#     MiniCNN, epochs=20, dataset=LayersCNNs2c6IG, batch_size=32
# ).train()

# Train(
#     MiniCNN, epochs=20, dataset=LayersCNNs3c6Grad, batch_size=32
# ).train()
#
# Train(
#     MiniCNN, epochs=20, dataset=LayersCNNs3c6SG, batch_size=32
# ).train()
#
# Train(
#     MiniCNN, epochs=20, dataset=LayersCNNs3c6IG, batch_size=32
# ).train()




#
# EvaluateModel(
#     MiniCNN,
#     "../models/MiniCNN_SplitClass10ResNets_Cifar10_Grad.h5",
#     SplitClass11thCifar10Grad,
# ).evaluate()

# EvaluateModel(
#     MiniCNN,
#     "../models/MiniCNN_SplitClass10ResNets_Cifar10_SG.h5",
#     SplitClass11thCifar10SG,
# ).evaluate()
#
# EvaluateModel(
#     MiniCNN,
#     "../models/MiniCNN_SplitClass10ResNets_Cifar10_IG.h5",
#     SplitClass11thCifar10IG,
# ).evaluate()


# EvaluateModel(
#     MiniCNN,
#     "../models/MiniCNN_SplitClass10ResNets_Cifar10_Grad.h5",
#     SplitClass11thCifar10TESTGrad,
# ).evaluate()
#
# EvaluateModel(
#     MiniCNN,
#     "../models/MiniCNN_SplitClass10ResNets_Cifar10_SG.h5",
#     SplitClass11thCifar10TESTSG,
# ).evaluate()
#
# EvaluateModel(
#     MiniCNN,
#     "../models/MiniCNN_SplitClass10ResNets_Cifar10_IG.h5",
#     SplitClass11thCifar10TESTIG,
# ).evaluate()









#################### OLD #################################################################




# GenerateSalDataset(
#     model=MiniResNetJ,
#     model_path="../models/MiniResNetJ_cifar10.h5",
#     dataset=Cifar10,
#     experiment_directory="testIDs",
# ).generate_dataset()

# Train(MiniCNN, epochs=20, dataset=ManyResNetsSG, batch_size=32).train()
#
# Train(MiniCNN, epochs=20, dataset=ManyResNetsIG, batch_size=32).train()


# GenerateSalDataset(
#     model=MiniResNetC,
#     model_path="../models/MiniResNetC_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()
#
# GenerateSalDataset(
#     model=MiniResNetD,
#     model_path="../models/MiniResNetD_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()
#
# GenerateSalDataset(
#     model=MiniResNetE,
#     model_path="../models/MiniResNetE_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()
#
# GenerateSalDataset(
#     model=MiniResNetF,
#     model_path="../models/MiniResNetF_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()
#
# GenerateSalDataset(
#     model=MiniResNetG,
#     model_path="../models/MiniResNetG_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()
#
# GenerateSalDataset(
#     model=MiniResNetH,
#     model_path="../models/MiniResNetH_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()
#
# GenerateSalDataset(
#     model=MiniResNetI,
#     model_path="../models/MiniResNetI_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()
#
# GenerateSalDataset(
#     model=MiniResNetJ,
#     model_path="../models/MiniResNetJ_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()

# EvaluateModel(
#     MiniCNN,
#     "../models/MiniCNN_ResA_ByClassCifar10_IG.h5",
#     ResTESTAByClassCifar10Grad,
# ).evaluate()
#
# EvaluateModel(
#     MiniCNN,
#     "../models/MiniCNN_ResA_ByClassCifar10_IG.h5",
#     ResTESTAByClassCifar10SG,
# ).evaluate()
#

#
# EvaluateModel(
#     MiniCNN,
#     "../models/MiniCNN_ResA_ByClassCifar10_SG.h5",
#     ResTESTBByClassCifar10SG,
# ).evaluate()
#
# EvaluateModel(
#     MiniCNN,
#     "../models/MiniCNN_ResA_ByClassCifar10_IG.h5",
#     ResTESTBByClassCifar10IG,
# ).evaluate()


#
# Train(
#     Small2CNN, epochs=20, dataset=ResAByClassCifar10SG, batch_size=32
# ).train()
#
# Train(
#     MiniCNN, epochs=20, dataset=ResAByClassCifar10SG, batch_size=32
# ).train()
#
# Train(
#     Small2CNN, epochs=20, dataset=ResAByClassCifar10IG, batch_size=32
# ).train()
#


# Train(
#     MiniResNetE, epochs=20, dataset=Cifar10, batch_size=32
# ).train()
#
# Train(
#     MiniResNetF, epochs=20, dataset=Cifar10, batch_size=32
# ).train()
#
# Train(
#     MiniResNetG, epochs=20, dataset=Cifar10, batch_size=32
# ).train()
#
# Train(
#     MiniResNetH, epochs=20, dataset=Cifar10, batch_size=32
# ).train()
#
# Train(
#     MiniResNetI, epochs=20, dataset=Cifar10, batch_size=32
# ).train()
#
# Train(
#     MiniResNetJ, epochs=20, dataset=Cifar10, batch_size=32
# ).train()
#


# #
# Train(CNN11L, epochs=300, dataset=Cifar10, batch_size=32).train()

# Train(CNN12L, epochs=50, dataset=Cifar10, batch_size=32, learning_rate=1e-4).train()


# Train(MiniCNN, epochs=20, dataset=LayersCNNs2c3, batch_size=32).train()
#
# Train(MiniResNet, epochs=20, dataset=LayersCNNs2c3, batch_size=32).train()
#
# # Train(Secondary2CNN, epochs=20, dataset=LayersCNNs11c12, batch_size=32).train()
#
# Train(Secondary2CNN, epochs=20, dataset=LayersCNNs11c12SG, batch_size=32).train()

# Train(Secondary2CNN, epochs=20, dataset=LayersCNNs2c3IG, batch_size=32).train()

# Train(MiniCNN, epochs=20, dataset=LayersCNNs11c12, batch_size=32).train()
# Train(MiniCNN, epochs=20, dataset=LayersCNNs11c12SG, batch_size=32).train()

# Train(Secondary2CNN, epochs=20, dataset=LayersCNNs2c3IG, batch_size=32).train()
# Train(MiniCNN, epochs=20, dataset=LayersCNNs11c12IG, batch_size=32).train()

# Train(Secondary2CNN, epochs=20, dataset=ResImageNetV2Grad, batch_size=32).train()

# Train(Secondary2CNN, epochs=10, dataset=ResImageNetV2SmoothGrad, batch_size=32).train()
# Train(Secondary2CNN, epochs=10, dataset=ResImageNetV2SmoothGrad, batch_size=32).train()
# Train(Secondary2CNN, epochs=10, dataset=ResImageNetV2IntegratedGrad, batch_size=32).train()

# Train(MiniCNN, epochs=20, dataset=ResImageNetV2Grad, batch_size=32).train()
# Train(MiniCNN, epochs=20, dataset=ResImageNetV2SmoothGrad, batch_size=32).train()
# Train(MiniCNN, epochs=20, dataset=ResImageNetV2IntegratedGrad, batch_size=32).train()
#

# # #

# class PretrainedGenerateSalDataset(GenerateSalDataset):
#     def load_model(self):
#         model = self.model.build()
#         model.summary()
#         return model
#
# class ResnetImageNetV2(ImageNetV2):
#     def augmentation(self, image, label, training=False):
#         image, label = super().augmentation(image, label, training)
#         image = tf.cast(image, tf.float32)
#         image = tf.keras.applications.resnet.preprocess_input(image)
#         return image, label
#
#
# # PretrainedGenerateSalDataset(
# #     model=ResNet101,
# #     model_path="",
# #     dataset=ResnetImageNetV2,
# #     saliency_method=None,
# # ).generate_examples()
# #
#
# PretrainedGenerateSalDataset(
#     model=ResNet50,
#     model_path="",
#     dataset=ResnetImageNetV2,
#     saliency_method=None,
# ).generate_examples()
#
# PretrainedGenerateSalDataset(
#     model=ResNet152,
#     model_path="",
#     dataset=ResnetImageNetV2,
#     saliency_method=None,
# ).generate_examples()


#

# GenerateSalDataset(
#     model=CNN2L,
#     model_path="../models/CNN_2L_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()


#
# GenerateSalDataset(
#     model=CNN3L,
#     model_path="../models/CNN_3L_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()
#
#
# GenerateSalDataset(
#     model=CNN4L,
#     model_path="../models/CNN_4L_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()
#
# GenerateSalDataset(
#     model=CNN5L,
#     model_path="../models/CNN_5L_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()
#
# GenerateSalDataset(
#     model=CNN6L,
#     model_path="../models/CNN_6L_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()
#
# GenerateSalDataset(
#     model=CNN11L,
#     model_path="../models/CNN_11L_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()
#
# GenerateSalDataset(
#     model=CNN12L,
#     model_path="../models/CNN_12L_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()


#
# GenerateSalDataset(
#     model=MiniCNNB,
#     model_path="../models/MiniCNNB_cifar10.h5",
#     dataset=Cifar10,
#     saliency_method=None,
# ).generate_dataset()


# EvaluateStandardized(StandRes2Cifar10Grad).evaluate()
