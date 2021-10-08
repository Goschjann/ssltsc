"""
Visualization functionalities
"""
import matplotlib
from matplotlib import rc
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import os
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import torchvision.transforms as transforms
import torchvision.utils as vutils
import tempfile

from ssltsc.constants import COLOR_MODEL_DICT, DATASET_NAMES_DICT, COLOR_MODEL_DICT_SLIDES, SUPERVISED_BASELINE
from pandas.plotting import table
from uncertainty_metrics.numpy.visualization import reliability_diagram

# Latex-ify plots
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


######
# In-use visualization functions for the paper
######

def plot_lines(path, dataset, storage_path='.', metric='test_weighted_auc', legend=False, store=True, xlab=False, ylab=False):
    data = pd.read_csv(f'{path}/{dataset}_aggregated_results.csv', header=[0, 1], index_col=[0, 1])
    print(f'loaded dataset {dataset}')
    # model_list = np.unique(data.index.get_level_values('model').values)
    model_list = ['ladder', 'logisticregression', 'meanteacher', 'mixmatch', 'randomforest', 'selfsupervised', 'supervised', 'vat']
    # from aistats results
    fully_supervised = 'fully_supervised' in model_list
    fully_supervised = True
    model_list = [model for model in model_list if model != 'fully_supervised']
    num_labels = np.unique(data.index.get_level_values('num_labels').values)
    num_labels = [int(num_label) for num_label in num_labels if num_label != 999]
    n_labels = len(num_labels)
    n_models = len(model_list)
    x = np.array([(i + 1) * 1500 for i in range(n_labels)])

    if store:
        plt.close()
    for i in range(len(model_list)):
        model_data = data.iloc[data.index.get_level_values('model') == model_list[i]]
        plt.errorbar(x= x + i * 100 - 300,
                     y=model_data[metric]['mean'],
                     yerr=model_data[metric]['std'], fmt='o',
                     color=COLOR_MODEL_DICT[model_list[i]]['color'],
                     linestyle=COLOR_MODEL_DICT[model_list[i]]['linestyle'])

    if fully_supervised:
        # mean_fully_labelled = data.iloc[data.index.get_level_values('model') == 'fully_supervised'][metric]['mean'].values[0]
        # from aistats results
        mean_fully_labelled = SUPERVISED_BASELINE[dataset]
        x[0] -= 300
        x[-1] += 300
        plt.plot(x, [mean_fully_labelled] * n_labels, linestyle='--', color='black')
        model_list.append('fully_supervised')
        n_models += 1
    plt.xlim(750, n_labels * 1500 + 750)
    plt.xticks(ticks=x, labels=num_labels)
    if ylab:
        plt.ylabel('weighted AUC')
    if xlab:
        plt.xlabel('Number of labelled samples')
    name_dataset = DATASET_NAMES_DICT[dataset]
    plt.title(f'{name_dataset}')
    if legend:
        plt.legend(prop={"size": 15}, handles=[mpatches.Patch(color=v['color'], label=v['name']) for (k, v) in COLOR_MODEL_DICT.items()])
        # plt.legend(handles=[mpatches.Patch(facecolor=v['color'], label=v['name'], hatch='OO') for (k, v) in COLOR_MODEL_DICT.items()])
    if store:
        plt.savefig(storage_path, dpi=1200)


def plot_lines_slides(path, dataset, storage_path='.', metric='test_weighted_auc', legend=False, store=True, xlab=False, ylab=False):

    data = pd.read_csv(f'{path}/{dataset}_aggregated_results.csv', header=[0, 1], index_col=[0, 1])

    model_list = np.unique(data.index.get_level_values('model').values)
    model_list = ['mixmatch', 'supervised', 'fully_supervised']
    fully_supervised = 'fully_supervised' in model_list
    model_list = [model for model in model_list if model != 'fully_supervised']
    num_labels = np.unique(data.index.get_level_values('num_labels').values)
    num_labels = [int(num_label) for num_label in num_labels if num_label != 999]
    n_labels = len(num_labels)
    n_models = len(model_list)
    x = np.array([(i + 1) * 1500 for i in range(n_labels)])

    if store:
        plt.close()
        plt.figure(figsize=(8, 4.5))
    for i in range(len(model_list)):
        model_data = data.iloc[data.index.get_level_values('model') == model_list[i]]
        plt.errorbar(x= x + i * 100 - 300,
                     y=model_data[metric]['mean'],
                     yerr=model_data[metric]['std'], fmt='o',
                     color=COLOR_MODEL_DICT[model_list[i]]['color'],
                     linestyle=COLOR_MODEL_DICT[model_list[i]]['linestyle'])

    if fully_supervised:
        mean_fully_labelled = data.iloc[data.index.get_level_values('model') == 'fully_supervised'][metric]['mean'].values[0]
        x[0] -= 300
        x[-1] += 300
        plt.plot(x, [mean_fully_labelled] * n_labels, linestyle='--', color='black')
        model_list.append('fully_supervised')
        n_models += 1
    plt.xlim(750, n_labels * 1500 + 750)
    plt.xticks(ticks=x, labels=num_labels)
    if ylab:
        plt.ylabel('Accuracy')
    if xlab:
        plt.xlabel('Number of labelled samples')
    plt.title('SITS Dataset')
    if legend:
        plt.legend(handles=[mpatches.Patch(color=v['color'], label=v['name']) for (k, v) in COLOR_MODEL_DICT_SLIDES.items()])
    if store:
        plt.savefig(storage_path, dpi=300)

def visualize_results_lineplot(mlflow_id=1,
                               performance_measures=None,
                               storage_path='../mlruns',
                               model_list=None):
    path = f'{storage_path}/{mlflow_id}'
    runs = pd.read_csv(f'{path}/results.csv')
    dataset = np.unique(runs['dataset'].values)[0]
    # de-select the tuning runs via the run tag
    runs = runs.loc[runs['run'] == 'training']
    # filter out fully labelled baseline to add it later to the aggregated data
    if sum(runs['num_labels'] > 1000) >= 1:
        fully_labelled_baseline = runs.loc[runs['num_labels'] > 1000]
        runs = runs.loc[runs['num_labels'] <= 1000]
    else:
        fully_labelled_baseline = None

    model_list = list(runs['model'].unique()) if model_list == None else model_list
    # model_list = [model for model in model_list if model != 'fully_labe']
    model_list.sort()
    performance_measures = [colname for colname in runs.columns.tolist() if colname.startswith('test')] if performance_measures == None else performance_measures

    # agg_data[agg_data['model'] == 'selfsupervised']
    agg_data = runs[['num_labels', 'model'] + performance_measures].round(decimals=5)
    agg_data = agg_data.groupby(['num_labels', 'model']).agg(['mean', 'std'])
    # add fully labelled baseline to the aggregated data
    if fully_labelled_baseline is not None:
        fully_labelled_baseline = fully_labelled_baseline[['num_labels', 'model'] + performance_measures].round(decimals=5)
        fully_labelled_baseline = fully_labelled_baseline.groupby(['num_labels', 'model']).agg(['mean', 'std'])
        # foo = [val for val in fully_labelled_baseline.values[0]]
        agg_data.loc[('999', 'fully_supervised'), :] = fully_labelled_baseline.values[0]

    agg_data.to_csv(f'{path}/{dataset}_aggregated_results.csv')

    # one line plot for each performance metric
    for metric in performance_measures:
        plot_lines(metric=metric,
                   path=path,
                   storage_path=f'{path}/{dataset}_performance_{metric}.png',
                   dataset=dataset,
                   legend=True,
                   store=True)
    print(f'Plotted results for data set {dataset}')


def store_reliability(y, yhat_prob, model_name=None):
    """Create and store a reliability plot to assess model calibration
    x-axis: binned model outputs
    y-axis: accuracy in each bin

    A perfectly calibrated model produces the diagonal line

    Code from https://github.com/google/uncertainty-metrics

    Args:
        y (ndarray): true label vector in format N
        yhat (ndarray): probability matrix in format N x Amount of Classes
    """
    diagram = reliability_diagram(labels=y, probs=yhat_prob, class_conditional=False)
    if model_name is not None:
        diagram.suptitle(f'Model: {model_name}')
    # store and log artifcat via tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage_path = os.path.join(tmp_dir, "reliability.png")
        diagram.savefig(storage_path, dpi=450)
        mlflow.log_artifact(local_path=storage_path)


#######
# Archived Functions that might become interesting again
#######



def plot_embed(embeddings, epoch):
    """ plot the replicated timeseries and the following embeddings
    derived from the LadderNet

    Args:
        embeddings: {list of list of torch.tensors} list of list of torch
            tensors containing the convolved feature maps
            from encoder and decoder
        epoch: {int} number of epoch
    Returns:
        {png} plt.plot that show the respective embeddings
    """
    embed_decoder = embeddings[0]
    embed_clean_encoder = embeddings[1]

    # levels: 4, 3, 2, 1 (=reconstructed input)
    dec_emb1 = embed_decoder[0][0, 0, :].cpu().detach().numpy()
    dec_emb2 = embed_decoder[1][0, 0, :].cpu().detach().numpy()
    dec_emb3 = embed_decoder[2][0, 0, :].cpu().detach().numpy()
    dec_emb4 = embed_decoder[3][0, 0, :].cpu().detach().numpy()

    # levels 1 (=original input), 2, 3, 4
    cenc_emb1 = embed_clean_encoder[0][0, 0, :].cpu().detach().numpy()
    cenc_emb2 = embed_clean_encoder[1][0, 0, :].cpu().detach().numpy()
    cenc_emb3 = embed_clean_encoder[2][0, 0, :].cpu().detach().numpy()
    cenc_emb4 = embed_clean_encoder[3][0, 0, :].cpu().detach().numpy()

    # equal ylims for all plots
    all_values = np.concatenate([dec_emb1, dec_emb2, dec_emb3, dec_emb4])
    ymin = min(all_values)
    ymax = max(all_values)

    plt.close()
    plt.figure(figsize=(8, 10))
    plt.suptitle("Epoch {}".format(epoch))

    # plot for decoder
    plt.subplot(421)
    plt.plot(dec_emb1)
    plt.title('Decoder - Level 4')
    plt.ylim(ymin, ymax)

    plt.subplot(423)
    plt.plot(dec_emb2)
    plt.title('Decoder - Level 3')
    plt.ylim(ymin, ymax)

    plt.subplot(425)
    plt.plot(dec_emb3)
    plt.title('Decoder - Level 2')
    plt.ylim(ymin, ymax)

    plt.subplot(427)
    plt.plot(dec_emb4)
    plt.title('Reproduced Input / Level 1')
    plt.ylim(ymin, ymax)

    # plot for clean encoder
    plt.subplot(422)
    plt.plot(cenc_emb4)
    plt.title('Clean Encoder - Level 4')
    plt.ylim(ymin, ymax)

    plt.subplot(424)
    plt.plot(cenc_emb3)
    plt.title('Clean Encoder - Level 3')
    plt.ylim(ymin, ymax)

    plt.subplot(426)
    plt.plot(cenc_emb2)
    plt.title('Clean Encoder - Level 2')
    plt.ylim(ymin, ymax)

    plt.subplot(428)
    plt.plot(cenc_emb1)
    plt.title('Original Input / Level 1')
    plt.ylim(ymin, ymax)

    plt.tight_layout()

    storage_path = 'Laddernet_embedding_{}.png'.format(epoch)
    plt.savefig(storage_path)
    mlflow.log_artifact(local_path=storage_path)
    os.system('rm Laddernet_embedding_{}.png'.format(epoch))


def plot_testacc_numlabels(dataset, models, results, path, suffix):
    """Scatter plot test accuracy vs. num labels for one data set over
        different models

    Arguments:
        dataset {str} -- the dataset to be plotted
        models {list} -- all models used in the experiment
        results {pd.DataFrame} -- dataframe with results
    """
    res = results[results['dataset'] == dataset]
    plt.close()
    for mod in models:
        res_model = res[res['model'] == mod]
        res_model.sort_values(by=['num_labels'], inplace=True)
        plt.scatter(res_model['num_labels'], res_model['test_acc'],
                    label=mod, marker='o')
    plt.ylabel('test accuracy')
    plt.xlabel('num labels')
    plt.title('Dataset: {}'.format(dataset))
    plt.legend()
    plt.tight_layout()
    if suffix is None:
        plt.savefig('{}{}_performance.png'.format(path, dataset))
    else:
        plt.savefig('{}{}_performance_{}.png'.format(path, dataset,
                                                     suffix))


def plot_performance_numlabels_std(dataset, models, path, alpha=0.2,
                                   measure='acc', suffix=None):
    res = pd.read_csv('{}/results_{}.csv'.format(path, dataset))
    plt.close()
    for mod in models:
        res.sort_values(by=['num_labels'], inplace=True)
        num_labels = res['num_labels'].astype(int).unique()
        mean = res[res['model'] == mod]['mean_{}'.format(measure)]
        stdev = res[res['model'] == mod]['std_{}'.format(measure)]
        plot_min = mean - stdev
        plot_max = mean + stdev
        plt.plot(num_labels, mean.values, label=mod, marker='o')
        plt.fill_between(num_labels, plot_min.values,
                         plot_max.values, alpha=alpha)
    plt.ylabel('test {}'.format(measure))
    plt.xlabel('amount of labels')
    plt.title('Dataset: {}'.format(dataset))
    plt.xticks(num_labels, num_labels)
    plt.legend()
    plt.tight_layout()
    if suffix is None:
        plt.savefig('{}{}_{}_stdev_performance.png'.format(path,
                                                           dataset,
                                                           measure))
    else:
        plt.savefig('{}{}_{}_stdev_performance_{}.png'.format(path,
                                                             dataset,
                                                             measure,
                                                             suffix))


def plot_mixup(X1, X2, X, Y1, Y2, Y, image=True, comment=None, storage_path='graphics/', dpi=300):
    # mkdir if not existing:
    if not os.path.exists(storage_path):
        os.mkdir(storage_path)
    if X1.ndim == 3:
        plt.close()
        plt.figure(figsize=(5, 16))

        # re-normalize the data for plotting
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
            )
        plt.subplot(3, 1, 1)
        img = np.uint8((inv_normalize(X1).cpu() * 255).numpy())
        # transform to format height, width, channels for plt
        plt.imshow(img.swapaxes(0, 1).swapaxes(1, 2))
        plt.title('{}\nInput 1 \n{}'.format(comment, Y1.tolist()))

        plt.subplot(3, 1, 2)
        img = np.uint8((inv_normalize(X2).cpu() * 255).numpy())
        # transform to format height, width, channels for plt
        plt.imshow(img.swapaxes(0, 1).swapaxes(1, 2))
        Y2 = [i.__round__(2) for i in Y2.tolist()]
        plt.title('Input 2 \n{}'.format(Y2))

        plt.subplot(3, 1, 3)
        img = np.uint8((inv_normalize(X).cpu() * 255).numpy())
        # transform to format height, width, channels for plt
        plt.imshow(img.swapaxes(0, 1).swapaxes(1, 2))
        Y = [i.__round__(2) for i in Y.tolist()]
        plt.title('Mixedup \n{}'.format(Y))

    else:
        plt.close()
        plt.figure(figsize=(16, 16))
        n_channels = X1.shape[0]
        for channel in range(n_channels):
            X_1 = X1[channel, :].cpu().detach().numpy()
            X_2 = X2[channel, :].cpu().detach().numpy()
            X_3 = X[channel, :].cpu().detach().numpy()

            all_values = np.concatenate([X_1, X_2, X_3])
            ymin = min(all_values)
            ymax = max(all_values)

            #plt.subplot(3, n_channels, 1 + 3*channel)
            #plt.subplot(3, n_channels*3, 1 + 3*channel)
            plt.subplot(n_channels, 3, 1 + 3*channel)
            plt.plot(X_1)
            if channel == 0:
                plt.title('{}\nInput 1 \n{}'.format(comment, Y1.tolist()))
            plt.ylim(ymin, ymax)

            #plt.subplot(3, n_channels, 2 + 3*channel)
            #plt.subplot(3, n_channels*3, 2 + 3*channel)
            plt.subplot(n_channels, 3, 2 + 3*channel)
            plt.plot(X_2)
            if channel == 0:
                Y_2 = [i.__round__(2) for i in Y2.tolist()]
                plt.title('Input 2 \n{}'.format(Y_2))
            plt.ylim(ymin, ymax)


            #plt.subplot(3, n_channels, 3 + 3*channel)
            #plt.subplot(3, n_channels*3, 3 + 3*channel)
            plt.subplot(n_channels, 3, 3 + 3*channel)
            plt.plot(X_3)
            if channel == 0:
                Y_mixed = [i.__round__(2) for i in Y.tolist()]
                plt.title('Mixedup \n{}'.format(Y_mixed))
            plt.ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig('{}mu_plot_{}.png'.format(storage_path, comment))


def plot_timeseries_aug(X, transform, suffix=None):
    """plot 8 augmented versions of one time series based on transform description

    Arguments:
        X {torch.tensor} -- torch tensor of one sample on cpu with shape: nchannels, tslength
        transform {torch.transforms} -- transforms decription

    Keyword Arguments:
        suffix {str} -- suffix for plot name (default: {None})
    """
    plt.close()
    plt.figure(figsize=(20, 16))
    plt.subplot(3, 3, 1)
    for ch in range(X.shape[0]):
        plt.plot(X[ch, :].numpy())
    plt.title('Original\n{}'.format(suffix))
    for i in range(1, 9):
        plt.subplot(3, 3, i + 1)
        for ch in range(X.shape[0]):
            plt.plot(transform(X[ch, :]).numpy())
    plt.tight_layout()
    if suffix is not None:
        plt.savefig('aug_plot_{}.png'.format(suffix))
    else:
        plt.savefig('aug_plot.png')

def plot_examples(XY, path, suffix=None, n_plots=5):
    """Plot examples for each class of a data set
    """
    n_classes = len(np.unique(XY[1]))
    cnt = 1
    plt.close()
    plt.figure(figsize=(3 * n_classes, 3 * n_plots))
    for j in range(n_plots):
        for i in range(n_classes):
            class_idx = np.where(XY[1] == i)[0]
            class_data = XY[0][class_idx, :, :]
            plt.subplot(n_plots, n_classes, cnt)
            plt.plot(class_data[j, 0, :])
            if j == 0:
                plt.title('Class {}'.format(i))
            cnt += 1
    plt.tight_layout()
    if suffix is not None:
        plt.savefig('{}sim_data_{}.png'.format(path, suffix))
    else:
        plt.savefig('{}sim_data.png'.format(path))

def plot_vat_examples(x, adv, y, path, suffix=None, n_plots=3):
    """Plot examples for each class of a data set
    """

    y = y.detach().cpu().softmax(1).numpy().argmax(1)
    x = x.detach().cpu().numpy()
    adv = adv.detach().cpu().numpy() + x

    # print(np.unique(y, return_counts=True))

    n_classes = len(np.unique(y))

    if min(np.unique(y, return_counts=True)[1]) < n_plots:
        n_plots = min(np.unique(y, return_counts=True)[1])

    cnt = 1
    plt.close()
    plt.figure(figsize=(3 * n_classes, 3 * n_plots))

    for j in range(n_plots):
        for i in range(n_classes):
            class_idx = np.where(y == i)[0]
            class_x = x[class_idx, :, :]
            class_adv = adv[class_idx, :, :]
            plt.subplot(n_plots, n_classes, cnt)
            plt.plot(class_adv[j, 0, :], ':', color='gray', label='x+radv')
            plt.plot(class_x[j, 0, :], '-', color='blue', label='x')
            if j == 0:
                plt.title('Predicted class {}'.format(i))
            cnt += 1
    plt.legend()
    plt.tight_layout()
    if suffix is not None:
        plt.savefig('{}adv_data_{}.png'.format(path, suffix), dpi=300)
    else:
        plt.savefig('{}adv_data.png'.format(path), dpi=300)
