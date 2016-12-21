'''
Created on 6 nov. 2015

@author: Faisal, Jos
'''
import operator
import os
import sys

import matplotlib
import numpy as np


def error_graph(names, avgs, stds, ns, xlabel, ylabel, filename):
    colors = ['blue', 'red', 'green']
    marks = ['*', 'square*', 'triangle*']

    tf = open("%s.tex" % filename, "w")
    tf.write("\\begin{figure}\n")
    tf.write("\\begin{tikzpicture}\n")
    tf.write("\\begin{axis}[\n")
    tf.write("\tgrid=major,\n")
    tf.write("\tymin=0,\n")
    tf.write("\txmin=0,\n")
    tf.write("\txmax=%i,\n" % ns[-1])
    tf.write("\txlabel={%s},\n" % xlabel)
    tf.write("\tylabel={%s},\n" % ylabel)
    tf.write("\tlegend style={at={(0.5,-0.17)},")
    tf.write("anchor=north,legend cell align=left}\n")
    tf.write("]\n")

    for col, mark in zip(colors, marks):
        tf.write("\\addlegendimage{mark=%s, %s}\n" % (mark, col))
    for i, name in enumerate(names):
        _name = name.replace(" ", "_")
        av = np.array(avgs[name])
        std = np.array(stds[name])
        low = map(operator.sub, av, std)
        hgh = map(operator.add, av, std)

        tf = _writeCoordinates(tf, _name, ns, av, legend=name, forget=False,
                               mark=marks[i])
        tf = _writeCoordinates(tf, '%s_lower_conf_bound' % _name, ns, low)
        tf = _writeCoordinates(tf, '%s_higher_conf_bound' % _name, ns, hgh)
    for i, name in enumerate(names):
        _name = name.replace(" ", "_")
        tf.write("\\addplot fill between[\n")
        tf.write("\tof = %s_lower_conf_bound and" % _name
                 + " %s_higher_conf_bound,\n" % _name)
        tf.write("\tevery even segment/.style = {%s!50}, forget plot,\n"
                 % colors[i])
        tf.write("];\n")
    tf.write("\\end{axis}\n")
    tf.write("\\end{tikzpicture}\n")
    tf.write("\\caption{Caption %s}\n" % ylabel)
    tf.write("\\label{fig:%s}\n" % ylabel)
    tf.write("\\end{figure}\n")
    tf.close()


def _writeCoordinates(tf, _name, ns, data, legend=None, mark=None,
                      forget=True):
    forget_txt = ', forget plot' if forget else ''
    mark_txt = ', mark=%s' % mark if mark is not None else ''
    tf.write("\\addplot[name path=%s, smooth%s%s] coordinates {\n"
             % (_name, forget_txt, mark_txt))
    for j, n in enumerate(ns):
        tf.write("\t(%i, %f)\n" % (n, data[j]))
    tf.write("};\n")
    if legend is not None:
        tf.write("\\addlegendentry{%s}\n" % legend)
    return tf


def ty(X, t, y, folder_results, fold=None, verbose=False):
    assert len(t) == len(y)
    data = np.vstack((t, y))
    diff = np.diff(data, axis=0)
    diff = diff.astype('float64')
    if type(diff[0]) != float:
        diff = diff[0]
    if fold is not None:
        print('\t\t\tFold %i' % fold),
    MAE = np.average(np.absolute(diff))
    RMSE = np.sqrt(np.average(diff ** 2))
    print '\tMAE: %f \t\tRMSE: %f' % (MAE, RMSE)
    if verbose:
        worst_pred = np.argmax(diff)
        print('\t\t\t\t\tWorst prediction: %.3f, should be %.3f.'
              % (y[worst_pred], t[worst_pred]))
        print('\t\t\t\t\tMean prediction: %.3f, mean real: %.3f'
              % (np.mean(y), np.mean(t)))
        print('\t\t\t\t\tStd prediction: %.3f, std real: %.3f'
              % (np.std(y), np.std(t)))
        print('\t\t\t\t\t# real is zero: %i'
              % int(t.shape[0] - np.count_nonzero(t)))

    _max = np.max(t)
    stepsize = _max / 50.
    _max += stepsize
    boxpl_pred, labels_pred = [], []
    for i in np.arange(0, _max, stepsize):
        indices = (i <= t) & (t < (i + stepsize))
        ys = list(y[indices].flat)
        if len(ys) > 0:
            boxpl_pred.append(ys)
            labels_pred.append('%.2f' % i)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(211, axisbg='w')
    plt.boxplot(boxpl_pred, labels=labels_pred)
    plt.ylabel('Prediction')
    plt.xlabel('Real value')
#         plt.ylim((0, 0.06))
    plt.subplot(212)
    plt.hist(t, bins=(_max / stepsize))
    plt.ylabel('# real value occurs')
    plt.xlabel('Real value')
    if not os.path.exists(folder_results):
        os.makedirs(folder_results)
    plt.savefig('%sdebuggraph%s.png'
                % (folder_results, '' if fold is None else fold))
    plt.close()
    sys.stdout.flush()
    return MAE, RMSE
