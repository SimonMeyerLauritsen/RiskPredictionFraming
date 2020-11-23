###################################################
#                IMPORT STATEMENTS                #
###################################################
import pandas as pd
import numpy as np
import shap as shap
import matplotlib.pylab as plt
import pickle
import xgboost as xgb

###################################################
#              READ DATA FROM DISK                #
###################################################
def read_data_from_disk(file_name):
    """
    Function to read data from disk

    :param file_name: file name
    :return: data (Pandas Dataframe)
    """
    with open(file_name, 'rb') as f:
        return pickle.load(f)
dataset_n = read_data_from_disk('FinalDatasets\dataset_Sepsis_n.pkl')[0]
dataset_d = read_data_from_disk('FinalDatasets\dataset_Sepsis_d.pkl')[0]
dataset_t = read_data_from_disk('FinalDatasets\dataset_Sepsis_t.pkl')[0]
dataset_r = read_data_from_disk('FinalDatasets\dataset_Sepsis_r.pkl')[0]

###################################################
#    VISUALIZE MISSING VALUES     #
###################################################
def plot_missing_values(df,fname):
    """
    Function to plot the missing values and save the result to disk

    :param df: data (Pandas Dataframe)
    :param fname: file name extension, when writing plot to disk
    """
    df = df.drop(columns=['DW_EK_Borger', 'label', 'sampleID'])
    m = pd.DataFrame(df.isnull().mean() * 100, columns=['percent_missing'])
    m.sort_values('percent_missing', inplace=True, ascending=False)
    m.to_excel(fname)
    ax = m.plot(kind='barh', figsize=(5, 10), zorder=2, width=0.5)
    ax.invert_yaxis()
    plt.savefig('Nybrud_Missing_Values.pdf', bbox_inches='tight', format='pdf', dpi=200)
    plt.show()

plot_missing_values(dataset_n, 'FinalDatasets\FinalResults\missing_n.xlsx')
plot_missing_values(dataset_d, 'FinalDatasets\FinalResults\missing_d.xlsx')
plot_missing_values(dataset_t, 'FinalDatasets\FinalResults\missing_t.xlsx')
plot_missing_values(dataset_r, 'FinalDatasets\FinalResults\missing_r.xlsx')

###################################################
#       SPLIT DATA INTO TRAIN AND TEST SETS       #
###################################################
def convert_raw_to_cv_folds(df):
    """
    Function to split det complete dataset into folds.

    :param df: Complete dataset (Pandas Dataframe)
    :return: List of data folds (Pandas Dataframe)
    """

    # create list of unique citizens (defines by the unique indentifier 'DW_EK_Borger'
    u, indices = np.unique(df['DW_EK_Borger'], return_index=True)

    # split unique citizens in five
    ps = np.asarray([list(t) for t in zip(*[iter(u)] * 5)])

    # split data in five based on citizen splits from above
    fold1 = df.loc[df['DW_EK_Borger'].isin(ps[:, 0])]
    fold2 = df.loc[df['DW_EK_Borger'].isin(ps[:, 1])]
    fold3 = df.loc[df['DW_EK_Borger'].isin(ps[:, 2])]
    fold4 = df.loc[df['DW_EK_Borger'].isin(ps[:, 3])]
    fold5 = df.loc[df['DW_EK_Borger'].isin(ps[:, 4])]

    def remove_meta(df):
        """
        Function to remove metadata from data

        :param df: Complete dataset with metadata (Pandas Dataframe)
        :return: Dataset without metadata (Pandas Dataframe)
        """
        y = df['label']
        x = df.loc[:, df.columns != 'label']
        x = x.loc[:, x.columns != 'DW_EK_Borger']
        x = x.loc[:, x.columns != 'sampleID']
        x = x.loc[:, x.columns != 'Sample_datotid_start']
        x = x.loc[:, x.columns != 'Sample_datotid_slut']
        return [x, y]

    fold1 = remove_meta(fold1)
    fold2 = remove_meta(fold2)
    fold3 = remove_meta(fold3)
    fold4 = remove_meta(fold4)
    fold5 = remove_meta(fold5)

    return [fold1, fold2, fold3, fold4, fold5]

folds_n = convert_raw_to_cv_folds(dataset_n)
folds_t = convert_raw_to_cv_folds(dataset_t)
folds_d = convert_raw_to_cv_folds(dataset_d)
folds_r = convert_raw_to_cv_folds(dataset_r)

###################################################
#          TRAIN RISK PREDICTION MODRELS          #
###################################################
def evaluate_classification_results(y_true, y_prob):
    """
    Function calculate AUROC and AUPRC performance values.

    :param y_true: Labels
    :param y_prob: Prediction labels
    :return: dictionary with AUROC and AUPRC performance results
    """

    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    average_precision = metrics.average_precision_score(y_true=y_true, y_score=y_prob)
    return {"Average precision": average_precision, "AUC": roc_auc}

def train_model(x_tr, y_tr, x_t, y_t, fname, nrounds, lr, md, l, mcw):
    """
    Function to built and train the sepsis risk prediction models with XGBoost

    :param x_tr: training features (Pandas Dataframe)
    :param y_tr: training labels (Pandas Series)
    :param x_t: test features (Pandas Dataframe)
    :param y_t: test labels (Pandas Series)
    :param fname: file name extension, when writing results to disk
    :param nrounds: number of boosting rounds
    :param lr: learning_rate
    :param md: max_depth
    :param l: lambda
    :param mcw: min_child_weight
    :return:
    """

    # convert to XGBoost format
    xgtrain = xgb.DMatrix(x_tr.values, y_tr.values)
    xgtest = xgb.DMatrix(x_t.values, y_t.values)

    # Initialize model parameters
    param = {'max_depth': md, 'objective': 'binary:logistic', 'eval_metric': 'aucpr',
             'learning_rate': lr, 'lambda': l, 'min_child_weight': mcw}
    num_round = nrounds

    # Initialize model
    model = xgb.train(param, xgtrain, num_boost_round=num_round, verbose_eval=True)

    # predict on test set based on trained model
    preds = model.predict(xgtest)

    # get performance metrics
    model_metrics = evaluate_classification_results(y_t, preds)

    # save results to disk
    pd.DataFrame.from_dict(model_metrics, orient='index').to_excel(fname)

    return preds, model_metrics, model

def cv_train(folds_n, fname='', nrounds=75, lr=0.15, md=7, l=8, mcw=7):
    """
    Function to train a model for each cross-valdation folds and combine results for explanation analysis

    :param folds_n: list of folds. Each fold consist of a Pandas Dataframe (features) and a Pandas series (labels)
    :param fname: file name extension, when writing results to disk
    :param nrounds: XGBoost - number of boosting rounds for
    :param lr: learning_rate
    :param md: max_depth
    :param l: lambda
    :param mcw: min_child_weight
    :return: Combined features for all folds 'x_train_list' and combined SHAP values for all folds 'shap_values_list'
    """

    # initialize empty lists
    x_train_list = []
    shap_values_list = []

    # loop over folds
    for r in range(0, 5):
        flist = [0, 1, 2, 3, 4]

        # seperate training from test data
        set1 = set(flist)
        set2 = set([r])
        train_folds = set1.difference(set2)
        test_folds = [r]

        i = 0
        x_train = pd.DataFrame([])
        y_train = pd.DataFrame([])
        x_test = pd.DataFrame([])
        y_test = pd.DataFrame([])

        # loop over fold data to create training and test data
        for f in folds_n:
            if i in train_folds:
                x_train = pd.concat([x_train, f[0]])
                y_train = pd.concat([y_train, f[1]])
            if i in test_folds:
                x_test = pd.concat([x_test, f[0]])
                y_test = pd.concat([y_test, f[1]])
            i = i+1

        # train XGBoost model on training data and test on test data.
        preds, model_metrics, model = train_model(x_train, y_train, x_test, y_test, 'FinalDatasets\FinalResults\metrics_'+fname+'_cv'+str(r)+'.xlsx', nrounds=nrounds, lr=lr, md=md, l=l, mcw=mcw)

        # do SHAP explanation analysis
        shap_values = explain_model(model, x_train, y_train, 'n', dpi=600, a=0.1, plot=0)

        # add data and SHAP results to list, such that combined SHAP analysis can be performed at later stage
        x_train_list.append(x_train)
        shap_values_list.append(shap_values)

    return pd.concat(x_train_list), np.vstack(shap_values_list)

total_n, shap_values_n = cv_train(folds_n, fname='n', nrounds=75, lr=0.15, md=7, l=8, mcw=7)
total_t, shap_values_t = cv_train(folds_t, fname='t', nrounds=75, lr=0.15, md=7, l=8, mcw=7)
total_d, shap_values_d = cv_train(folds_d, fname='d', nrounds=75, lr=0.15, md=7, l=8, mcw=7)
total_r, shap_values_r = cv_train(folds_r, fname='r', nrounds=75, lr=0.15, md=7, l=8, mcw=7)

###################################################
#         USE SHAP TOOLBOX FOR SHAP VALUES        #
###################################################
def explain_model(model, x_t, y_t, fname, plot=False, run_shap=True, dpi=100, a=0.1, shap_values=[]):
    """
    Function to run SHAP explanation anlysis and save plots to disk af pdf files

    :param model: trained XGBoost model
    :param x_t: feature matrix
    :param y_t: labels
    :param fname: file name
    :param plot: parameter to control what to plot
    :param run_shap: parameter that controls to run shap or not
    :param dpi: dpi for pdf file
    :param a: controls the alpha parameter in the shap summary plot
    :param shap_values: shap_values. Only used if run_shap=True
    :return:
    """
    plt.clf()
    if run_shap == True:

        # run SHAP analysis
        xgtrain = xgb.DMatrix(x_t.values, y_t.values)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(xgtrain)

    if plot == 0:

        # return SHAP values and do not plot
        return shap_values

    if plot == 1 or plot == 2:

        # bar plots with SHAP summary_plot
        shap.summary_plot(shap_values, x_t, plot_type="bar", show=False, max_display=10)
        plt.savefig('FinalDatasets\FinalResults\shap_bar_10'+fname+'.pdf', bbox_inches='tight', format='pdf', dpi=dpi)
        plt.clf()
        shap.summary_plot(shap_values, x_t, plot_type="bar", show=False, max_display=50)
        plt.savefig('FinalDatasets\FinalResults\shap_bar_50_' + fname + '.pdf', bbox_inches='tight', format='pdf', dpi=dpi)
        plt.clf()

    if plot == 2:

        # dot plots with SHAP summary_plot
        shap.summary_plot(shap_values, x_t, show=False, alpha=a, plot_type='dot', layered_violin_max_num_bins=20, max_display=10)
        plt.savefig('FinalDatasets\FinalResults\shap_dot_10_'+fname+'.pdf', bbox_inches='tight', format='pdf', dpi=dpi, )
        plt.clf()
        shap.summary_plot(shap_values, x_t, show=False, alpha=a, plot_type='dot', layered_violin_max_num_bins=20, max_display=50)
        plt.savefig('FinalDatasets\FinalResults\shap_dot_50_' + fname + '.pdf', bbox_inches='tight', format='pdf', dpi=dpi, )
        plt.clf()

    return shap_values

explain_model(model=[], x_t=total_n, y_t=[], fname='n', dpi=600, a=0.1, plot=2, run_shap=False, shap_values=shap_values_n)
explain_model(model=[], x_t=total_t, y_t=[], fname='t', dpi=600, a=0.1, plot=2, run_shap=False, shap_values=shap_values_t)
explain_model(model=[], x_t=total_d, y_t=[], fname='d', dpi=600, a=0.1, plot=2, run_shap=False, shap_values=shap_values_d)
explain_model(model=[], x_t=total_r, y_t=[], fname='r', dpi=600, a=0.1, plot=2, run_shap=False, shap_values=shap_values_r)

###################################################
#              PLOT SHAP DEPENDENCE               #
###################################################



def plot_dependence(shap_values_n, shap_values_t, shap_values_d, shap_values_r, fname, xmi=0, xma=100, x_train_n=[],
                    x_train_t=[], x_train_d=[], x_train_r=[]):
    """
    Function to plot SHAP dependence plots and save to disk

    :param shap_values_n:
    :param shap_values_t:
    :param shap_values_d:
    :param shap_values_r:
    :param fname: file name extension, when writing results to disk
    :param xmi:
    :param xma:
    :param x_train_n:
    :param x_train_t:
    :param x_train_d:
    :param x_train_r:
    :return:
    """

    plt.clf()
    shap.dependence_plot(fname, shap_values_n, x_train_n.values, feature_names=x_train_n.columns,
                         interaction_index=None, xmin=xmi, xmax=xma, x_train_n=x_train_n, x_train_t=x_train_t,
                         x_train_d=x_train_d, x_train_r=x_train_r)
    plt.savefig('FinalDatasets\FinalResults\shap_' + fname + '_n.pdf', bbox_inches='tight', format='pdf', dpi=600)
    plt.clf()
    shap.dependence_plot(fname, shap_values_t, x_train_t.values, feature_names=x_train_t.columns,
                         interaction_index=None, xmin=xmi, xmax=xma, x_train_n=x_train_n, x_train_t=x_train_t,
                         x_train_d=x_train_d, x_train_r=x_train_r)
    plt.savefig('FinalDatasets\FinalResults\shap_' + fname + '_t.pdf', bbox_inches='tight', format='pdf', dpi=600)
    plt.clf()
    shap.dependence_plot(fname, shap_values_d, x_train_d.values, feature_names=x_train_d.columns,
                         interaction_index=None, xmin=xmi, xmax=xma, x_train_n=x_train_n, x_train_t=x_train_t,
                         x_train_d=x_train_d, x_train_r=x_train_r)
    plt.savefig('FinalDatasets\FinalResults\shap_' + fname + '_d.pdf', bbox_inches='tight', format='pdf', dpi=600)
    plt.clf()
    shap.dependence_plot(fname, shap_values_r, x_train_r.values, feature_names=x_train_r.columns,
                         interaction_index=None, xmin=xmi, xmax=xma,x_train_n=x_train_n, x_train_t=x_train_t,
                         x_train_d=x_train_d, x_train_r=x_train_r)
    plt.savefig('FinalDatasets\FinalResults\shap_' + fname + '_r.pdf', bbox_inches='tight', format='pdf', dpi=600)
    plt.clf()

plot_dependence(shap_values_n, shap_values_t, shap_values_d, shap_values_r, 'Temperature', xmi=32, xma=42)
plot_dependence(shap_values_n, shap_values_t, shap_values_d, shap_values_r, 'Heart rate', xmi=30, xma=200)
plot_dependence(shap_values_n, shap_values_t, shap_values_d, shap_values_r, 'Respiratory Frequency', xmi=0, xma=50)
plot_dependence(shap_values_n, shap_values_t, shap_values_d, shap_values_r, 'SpO2', xmi=75, xma=100)
plot_dependence(shap_values_n, shap_values_t, shap_values_d, shap_values_r, 'Diastolic BP', xmi=20, xma=150)
plot_dependence(shap_values_n, shap_values_t, shap_values_d, shap_values_r, 'Systolic BP', xmi=50, xma=250)
plot_dependence(shap_values_n, shap_values_t, shap_values_d, shap_values_r, 'B-Leukocytes', xmi=0, xma=30)
plot_dependence(shap_values_n, shap_values_t, shap_values_d, shap_values_r, 'B-Platelets', xmi=0, xma=1000)
plot_dependence(shap_values_n, shap_values_t, shap_values_d, shap_values_r, 'P-Glucose', xmi=2, xma=30)
plot_dependence(shap_values_n, shap_values_t, shap_values_d, shap_values_r, 'P-Creatinine', xmi=0, xma=800)
plot_dependence(shap_values_n, shap_values_t, shap_values_d, shap_values_r, 'P-Sodium', xmi=100, xma=180)
plot_dependence(shap_values_n, shap_values_t, shap_values_d, shap_values_r, 'P-Potassium', xmi=1, xma=8)