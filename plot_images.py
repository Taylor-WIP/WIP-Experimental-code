import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf


# models = ["MiniCNN", "MiniResNet", "MLP", "Small2CNN", "Small3CNN"]
models = ["MiniResNet", "MiniResNetB"]
image_titles = ["Coat", "Coat", "Ankle Boot", "Sneaker", "Sandal"]
image_class = [4, 4, 9, 7, 5]
#
methods = ["grad", "smoothGrad", "integratedGrad"]


subplot_args = {
    "nrows": 10,
    "ncols": 3,
    "figsize": (15, 50),
    "subplot_kw": {"xticks": [], "yticks": []},
}


f, ax = plt.subplots(**subplot_args)

count = 1
for model in models:
    ax[0][count].set_ylabel("{}".format(model), fontsize=45)
    for i in range(0, 10):
        # ax[count][i].set_title(classification, fontsize=30)
        ax[i][count].imshow(
            np.array(
                load_img(
                    "../images/cifar10/res_examples/integratedGrad/{}_{}.png".format(
                        i, model
                    )
                )
            )
        )

    count += 1

for i in range(0, 10):
    # ax[0][i].set_title(title, fontsize=40)
    ax[i][0].imshow(
        np.array(load_img("../images/cifar10/res_examples/{}_image.png".format(i)))
    )


plt.tight_layout()
plt.savefig("../images/cifar10/res_examples/res2_integratedGrad_example.png")

# im1 = load_img("../images/FashionMnist/examples/grad/0_Small2CNN_4.png")
# im2 = load_img("../images/FashionMnist/examples/grad/0_Small3CNN_4.png")

# def ssim_single(image_id):
#     paths = []
#     for model in models:
#         image_path = "../images/FashionMnist/examples/grad/{}_{}_{}.png".format(image_id, model, image_class[image_id])
#         paths.append(image_path)
#
#     results_list = []
#
#     for i in range(0, len(paths)):
#         im1=load_img(paths[i])
#         image_list=[]
#         for j in range(0, len(paths)):
#             im2=load_img(paths[j])
#             ssim = (tf.image.ssim(im1, im2, 1).numpy())
#             image_list.append(ssim)
#         results_list.append(image_list)
#
#     return results_list
#
#
# def average_ssim(ssim_list):
#     av_list = []
#     for i in range(0, len(models)):
#         partial_list = []
#         for j in range(0, len(models)):
#             current_comparison = []
#             for comp in ssim_list:
#                 current_comparison.append(comp[i][j])
#             average = np.mean(current_comparison)
#             partial_list.append(average)
#
#         av_list.append(partial_list)
#     return av_list
#
# all_ssim = []
# for i in range(0, 5):
#     all_ssim.append(ssim_single(i))
#
# print(average_ssim(all_ssim))
