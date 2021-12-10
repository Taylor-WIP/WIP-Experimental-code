import tensorflow as tf
import os
import re
import tqdm

#
# class SplitDataset:
#     def load_image(image_path):
#         return tf.io.decode_png(tf.io.read_file(image_path))
#
#     def split_class(image_name):
#         label = image_name[-5]
#         return label


# directory = "../images/cifar10/gradient/ResNets/Training/MiniResNet"
#
# files = os.listdir(directory)


############ ImageNetV2 Version ############


# function which splits image name to lable

# labels = []
# for n in range(1000):
#     labels.append(n)
#
# nets = ["50","101", "152"]
# methods = ["gradient", "smoothGrad", "integratedGrad"]
#
# for method in methods:
#     for net in nets:
#         for label in labels:
#             new_dirrectory = "../images/imagenet_v2/splitByClass/{}/ResNet{}/{}".format(
#             method, net, str(label))
#             if not os.path.exists(new_dirrectory):
#                 os.makedirs(new_dirrectory)


# methods = ["gradient", "smoothGrad", "integratedGrad"]
# networks = ["ResNet50", "ResNet101", "ResNet152"]
# for method in methods:
#     print(method)
#     for network in networks:
#         directory = "../images/imagenet_v2/{}/{}".format(method, network)
#         files = os.listdir(directory)
#
#         print(network)
#         count = 0
#         for file in tqdm.tqdm(files):
#             file_name = os.path.split(file)[-1]
#
#             split_name = re.split("[_.]", file_name)
#             label = split_name[1]
#             test = split_name[0]
#
#             # print(test, label)
#             # if count == 10:
#             #     break
#             # count+=1
#
#             save_path = "../images/imagenet_v2/splitByClass/{}/{}/{}".format(
#                 method, network, str(label)
#             )
#
#             os.system("cp {}/{} {}/".format(directory, file_name, save_path))


######## CIFAR10 VERSION ##############
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# networks = [
#     "MiniResNet",
#     "MiniResNetB",
#     "MiniResNetC",
#     "MiniResNetD",
#     "MiniResNetE",
#     "MiniResNetF",
#     "MiniResNetG",
#     "MiniResNetH",
#     "MiniResNetI",
#     "MiniResNetJ",
# ]

networks = ["MiniResNetK"]

methods = ["gradient", "smoothGrad", "integratedGrad"]

sets = ["Test_set", "Training"]

# for method in methods:
#     for set in sets:
#         # for network in networks:
#         for label in labels:
#             new_dirrectory = (
#                 "../images/cifar10/{}/ResNets/splitClassManyResNets/EleventhResNet/{}/{}".format(
#                     method, set, str(label)
#                 )
#             )
#             if not os.path.exists(new_dirrectory):
#                 os.makedirs(new_dirrectory)


# methods = ["gradient", "smoothGrad", "integratedGrad"]
# networks = [
#     "MiniResNet",
#     "MiniResNetB",
#     "MiniResNetC",
#     "MiniResNetD",
#     "MiniResNetE",
#     "MiniResNetF",
#     "MiniResNetG",
#     "MiniResNetH",
#     "MiniResNetI",
#     "MiniResNetJ",
# ]
#
# sets = ["Test_set", "Training"]
#
# for method in methods:
#     for set in sets:
#         for network in networks:
#             directory = "../images/cifar10/{}/ResNets/ManyResNets/{}/{}".format(
#                 method, set, network
#             )
#             files = os.listdir(directory)
#
#             for file in tqdm.tqdm(files):
#                 file_name = os.path.split(file)[-1]
#                 save_name = "{}_{}".format(network, file_name)
#                 label = file_name[-5]
#                 save_path = "../images/cifar10/{}/ResNets/splitClassManyResNets/TenResNets/{}/{}".format(
#                     method, set, str(label)
#                 )
#                 os.system("cp {}/{} {}/".format(directory, file_name, save_path))
#                 os.system(
#                     "mv {}/{} {}/{}".format(save_path, file_name, save_path, save_name)
#                 )
# #

for method in methods:
    for set in sets:
        for network in networks:
            directory = "../images/cifar10/{}/ResNets/ManyResNets/EleventhResNet/{}/{}".format(
                method, set, network
            )
            files = os.listdir(directory)

            for file in tqdm.tqdm(files):
                file_name = os.path.split(file)[-1]
                save_name = "{}_{}".format(network, file_name)
                label = file_name[-5]
                save_path = "../images/cifar10/{}/ResNets/splitClassManyResNets/EleventhResNet/{}/{}".format(
                    method, set, str(label)
                )
                os.system("cp {}/{} {}/".format(directory, file_name, save_path))
                os.system(
                    "mv {}/{} {}/{}".format(save_path, file_name, save_path, save_name)
                )





####### FOR VM LAYERS DATASET - move and arange DATASETS
#
# methods = ["gradient", "smoothGrad", "integratedGrad"]
# sets = ["Test_set", "Training"]
#
# networks = ["CNN_2L", "CNN_3L", "CNN_4L", "CNN_5L", "CNN_6L"]
#
# experiments = ["cnns_4_5", "cnns_5_6", "cnns_2_6", "cnns_3_6"]

#"cnns_3_4"]
#
# for method in methods:
#     for set in sets:
#         # for network in networks:
#         for experiment in experiments:
#             new_dirrectory = (
#                 "../images/cifar10/{}/{}/{}".format(
#                     method, set, experiment
#                 )
#             )
#             if not os.path.exists(new_dirrectory):
#                 os.makedirs(new_dirrectory)



# for method in methods:
#     for set in sets:
#         # for network in networks:
#         for experiment in experiments:
#             new_dirrectory = (
#                 "../images/cifar10/{}/{}/{}".format(
#                     method, set, experiment
#                 )
#             )
#             network1 = experiment[5]
#             network2 = experiment[7]
#
#             for network in networks:
#                 if set == "Test_set":
#                     net_location = "../images/cifar10/{}/{}/{}".format(method, set, network)
#                 else:
#                     net_location = "../images/cifar10/{}/{}/all7_cnns/{}".format(method, set, network)
#                 netID = network[4]
#                 if netID == network1 or netID == network2:
                    # os.system("cp -r {} {}/".format(net_location, new_dirrectory))
