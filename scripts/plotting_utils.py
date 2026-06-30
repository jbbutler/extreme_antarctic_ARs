'''
Script with some common plotting functions used in the model interpretations notebooks.

Jimmy Butler
12/2025
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_variable_importance(importances, ax, title='', value_col='avg_importance_scaled', scale=100,
    xlabel='Decrease in Predictive R$^{2}$, Scaled', xlim=(0, 115), normalize=False, tick_labels=None,
    label_fontsize=12, color=None, label_fmt='{:.2f}'):
    '''
    Plot scaled variable importance as a horizontal bar chart on a given axis.
    Inputs:
        importances (pd.DataFrame): Importance data, indexed by feature name.
        ax (matplotlib.axes.Axes): Axis to draw on.
        title (str): Plot title.
        value_col (str): Column in `importances` holding the importance values.
        scale (float): Multiplier applied to the values (e.g. 100 to convert to percent).
        xlabel (str): X-axis label.
        xlim (tuple or None): X-axis limits; pass None to let matplotlib auto-scale.
        normalize (bool): should we normalize by the highest permutation score?
        tick_labels (list of str or None): Replacement labels, given in the SAME order as the
            DataFrame index (reordered internally to match the sorted bars). If None, the
            index is used as-is.
        label_fontsize (int): Font size for the bar value labels.
        color (matplotlib color or None): Bar color (None = matplotlib default, matching
            plot_shap_summary_bar).
        label_fmt (str): Format string for the value annotation on each bar.
    Outputs
        matplotlib.axes.Axes
    '''
    names = list(importances.index)
    values = importances[value_col].to_numpy()
    if normalize:
        values = values / values.max()
    values = np.round(values, 4)
    values = values * scale

    display = list(tick_labels) if tick_labels is not None else names
    order = np.argsort(values)  # ascending -> largest at top with barh
    vals = values[order]
    labels = [display[i] for i in order]
    y = np.arange(len(vals))

    ax.barh(y, vals, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('')
    if xlim is not None:
        ax.set_xlim(xlim)

    for yi, v in zip(y, vals):
        ax.text(v, yi, ' ' + label_fmt.format(v),
                va='center', ha='left', fontsize=label_fontsize)

    return ax
 
def plot_pdp_1d(grid, pdp_vals, ax, feature_values=None, title='', xlabel='',
    ylabel='Partial Dependence', xlim=None, rug=True, rug_kwargs=None, line_kwargs=None):
    '''
    Plot a single 1D partial dependence curve on a given axis.
    Inputs:
        grid (array-like): Feature values at which the PDP was evaluated (x-axis).
        pdp_vals (array-like): Partial dependence values (y-axis), same length as `grid`.
        ax (matplotlib.axes.Axes): Axis to draw on.
        feature_values (array-like or None): Raw observed values of the feature, used to
            draw a rug. If None, no rug is drawn.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        xlim (tuple or None): X-axis limits; if None, set to (min(grid), max(grid)).
        rug (bool): Whether to draw a rugplot of `feature_values`.
        rug_kwargs (dict or None): Extra kwargs forwarded to sns.rugplot.
        line_kwargs (dict or None): Extra kwargs forwarded to ax.plot.
    Outputs
        matplotlib.axes.Axes
    '''
    grid = np.asarray(grid)
 
    ax.plot(grid, pdp_vals, **(line_kwargs or {}))
 
    if rug and feature_values is not None:
        rk = dict(height=0.05, color='gray', alpha=0.05)
        rk.update(rug_kwargs or {})
        sns.rugplot(x=feature_values, ax=ax, **rk)
 
    if xlim is None and grid.size:
        xlim = (grid.min(), grid.max())
    if xlim is not None:
        ax.set_xlim(xlim)
 
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax
 
def plot_pdp_2d(grid1, grid2, pdp_matrix, ax, feat_x_values=None, feat_y_values=None,
    title='', xlabel='', ylabel='', cmap='YlGnBu', levels=25, kde=True, kde_kwargs=None):
    '''
    Plot a single 2D partial dependence contour on a given axis.
    Inputs:
        grid1 (array-like): 1D grid for the x feature.
        grid2 (array-like): 1D grid for the y feature.
        pdp_matrix (2D array-like): Partial dependence surface, shape
            (len(grid2), len(grid1)) as produced by the meshgrid-style compute_pdp_2d.
        ax (matplotlib.axes.Axes): Axis to draw on.
        feat_x_values (array-like or None): Raw observed values for the x feature, used
            to overlay a KDE of the observed joint distribution.
        feat_y_values (array-like or None): Raw observed values for the y feature; both
            feat_x_values and feat_y_values must be provided to draw the KDE.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        cmap (str): Colormap; use 'YlGnBu' for snow, 'YlOrRd' for temperature.
        levels (int): Number of contour levels.
        kde (bool): Whether to overlay a KDE of the observed data.
        kde_kwargs (dict or None): Extra kwargs forwarded to sns.kdeplot.
    Outputs
        (matplotlib.axes.Axes, matplotlib.contour.QuadContourSet): The axis and the
            filled-contour set (useful for adding a colorbar).
    '''
    cf = ax.contourf(grid1, grid2, pdp_matrix, cmap=cmap, levels=levels)
 
    if kde and feat_x_values is not None and feat_y_values is not None:
        kk = dict(levels=8, color='black', linewidths=0.5, fill=False)
        kk.update(kde_kwargs or {})
        sns.kdeplot(x=feat_x_values, y=feat_y_values, ax=ax, **kk)
 
    ax.set_xlim(np.min(grid1), np.max(grid1))
    ax.set_ylim(np.min(grid2), np.max(grid2))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax, cf
 
def _resolve_shap_column(shap_values, feature, feature_names=None):
    '''
    Return the 1D SHAP value vector for a single feature.
    Inputs:
        shap_values (pd.DataFrame or 2D np.ndarray): Per-observation, per-feature SHAP values.
        feature (str): Feature (column) to extract.
        feature_names (list or None): Required if `shap_values` is a plain array.
    Outputs
        np.ndarray
    '''
    if isinstance(shap_values, pd.DataFrame):
        return shap_values[feature].to_numpy()
    sv = np.asarray(shap_values)
    if feature_names is None:
        raise ValueError("feature_names is required when shap_values is an array.")
    idx = list(feature_names).index(feature)
    return sv[:, idx]
 
 
def _resolve_feature_column(feature_values, feature, feature_names=None):
    '''
    Return the 1D observed-value vector for a single feature.
    Inputs:
        feature_values (pd.DataFrame or 2D np.ndarray): Observed feature values.
        feature (str): Feature (column) to extract.
        feature_names (list or None): Required if `feature_values` is a plain array.
    Outputs
        np.ndarray
    '''
    if isinstance(feature_values, pd.DataFrame):
        return feature_values[feature].to_numpy()
    fv = np.asarray(feature_values)
    if feature_names is None:
        raise ValueError("feature_names is required when feature_values is an array.")
    idx = list(feature_names).index(feature)
    return fv[:, idx]
 
 
def plot_shap_dependence(shap_values, feature_values, feature, ax, feature_names=None,
    title='', xlabel='', ylabel='SHAP Value', zero_line=True, scatter_kwargs=None):
    '''
    SHAP dependence scatter for one feature: observed value (x) vs SHAP value (y).
    Equivalent to shap.dependence_plot(..., interaction_index=None) but takes the raw
    SHAP array directly.
    Inputs:
        shap_values (pd.DataFrame or 2D np.ndarray): Per-observation, per-feature SHAP values.
        feature_values (pd.DataFrame or 2D np.ndarray): Observed feature values, aligned
            row-for-row with `shap_values`.
        feature (str): Feature (column) to plot.
        ax (matplotlib.axes.Axes): Axis to draw on.
        feature_names (list or None): Required only if `shap_values`/`feature_values` are
            plain arrays.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        zero_line (bool): Draw a dashed horizontal line at SHAP = 0.
        scatter_kwargs (dict or None): Extra kwargs forwarded to ax.scatter.
    Outputs
        matplotlib.axes.Axes
    '''
    sv = _resolve_shap_column(shap_values, feature, feature_names)
    x = _resolve_feature_column(feature_values, feature, feature_names)
 
    sk = dict(s=20, alpha=0.6, edgecolor='none')
    sk.update(scatter_kwargs or {})
    ax.scatter(x, sv, **sk)
 
    if zero_line:
        ax.axhline(0, color='gray', lw=0.8, ls='--')
 
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax
 
 
def plot_shap_summary_bar(shap_values, ax, feature_names=None, tick_labels=None, sort=True, title='',
    xlabel='mean(|SHAP value|)', color=None, label_fmt='{:.3f}', label_fontsize=11, pad_frac=0.15,
    normalize=False, scale=100):
    '''
    Global SHAP importance: mean absolute SHAP value per feature as a bar chart.
    Equivalent to shap.summary_plot(..., plot_type="bar") but takes the raw SHAP array
    directly.
    Inputs:
        shap_values (pd.DataFrame or 2D np.ndarray): Per-observation, per-feature SHAP values.
        ax (matplotlib.axes.Axes): Axis to draw on.
        feature_names (list or None): Required if `shap_values` is a plain array; otherwise
            the DataFrame columns are used.
        tick_labels (list of str or None): Replacement labels, given in the SAME order as
            `feature_names` / columns (they are reordered internally to match the sorted bars).
        sort (bool): Sort by mean(|SHAP|); largest ends up at the top.
        xlabel (str): X-axis label.
        color (matplotlib color or None): Bar color.
        label_fmt (str or None): Format string for the value annotation on each bar; pass
            None to omit. Ignored when normalize=True (values are shown as whole numbers).
        label_fontsize (int): Font size for the bar value labels.
        pad_frac (float): Fraction of the largest bar added as right-hand headroom so the
            largest bar's value label isn't clipped.
        normalize (bool): if True, express each bar as a fraction of the largest mean(|SHAP|),
            multiplied by `scale`, and rounded to the nearest whole number.
        scale (float): multiplier applied when normalize=True (e.g. 100 for a 0-100 scale).
    Outputs
        matplotlib.axes.Axes
    '''
    if isinstance(shap_values, pd.DataFrame):
        names = list(shap_values.columns)
        mean_abs = shap_values.abs().mean(axis=0).to_numpy()
    else:
        sv = np.asarray(shap_values)
        if feature_names is None:
            raise ValueError("feature_names is required when shap_values is an array.")
        names = list(feature_names)
        mean_abs = np.abs(sv).mean(axis=0)

    if normalize:
        mean_abs = mean_abs / mean_abs.max()
        mean_abs = np.round(mean_abs, 4)
        mean_abs = mean_abs * scale
    else:
        mean_abs = np.round(mean_abs, 2)

    display = list(tick_labels) if tick_labels is not None else names
    order = np.argsort(mean_abs)
    if not sort:
        order = np.arange(len(mean_abs))

    vals = mean_abs[order]
    labels = [display[i] for i in order]
    y = np.arange(len(vals))

    ax.barh(y, vals, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, vals.max() * (1 + pad_frac))

    if label_fmt is not None:
        for yi, v in zip(y, vals):
            ax.text(v, yi, ' ' + label_fmt.format(v),
                    va='center', ha='left', fontsize=label_fontsize)

    return ax
 
 
def plot_shap_waterfall(shap_row, base_value, ax, feature_values=None, feature_names=None,
    max_display=10, title='', pos_color='#ff0051', neg_color='#008bfb', value_fmt='{:+.2f}', annotate=True):
    '''
    Waterfall plot of a single observation's SHAP values.
    Shows how each feature moves the prediction from the base value (E[f(X)]) to the final
    prediction f(x). Features are ordered by absolute contribution; the largest sits at the
    top and any beyond `max_display` are collapsed into one "other features" bar.
    Inputs:
        shap_row (pd.Series or 1D array-like): SHAP values for ONE observation
            (length = n_features).
        base_value (float): The explainer's expected value, E[f(X)].
        ax (matplotlib.axes.Axes): Axis to draw on.
        feature_values (pd.Series or 1D array-like or None): Observed feature values for this
            same observation; if given, they are prepended to each label as "value = name".
        feature_names (list or None): Required if `shap_row` is a plain array (and not a Series).
        max_display (int): Maximum number of individual feature bars before lumping the rest.
        title (str): Plot title.
        pos_color (matplotlib color): Color for positive contributions.
        neg_color (matplotlib color): Color for negative contributions.
        value_fmt (str or None): Format for the per-bar value annotation; pass None to omit.
        annotate (bool): Whether to annotate each bar with its value.
    Outputs
        matplotlib.axes.Axes
    '''
    if isinstance(shap_row, pd.Series):
        names = list(shap_row.index)
        vals = shap_row.to_numpy(dtype=float)
    else:
        vals = np.asarray(shap_row, dtype=float)
        if feature_names is None:
            raise ValueError("feature_names is required when shap_row is an array.")
        names = list(feature_names)
 
    # Prepend observed values to labels if provided.
    if feature_values is not None:
        fv = (feature_values.to_numpy() if isinstance(feature_values, pd.Series)
              else np.asarray(feature_values))
        names = [f'{v:.2f} = {n}' for n, v in zip(names, fv)]
 
    # Order by descending |SHAP|.
    order = np.argsort(np.abs(vals))[::-1]
    vals_d = vals[order]
    names_d = [names[i] for i in order]
 
    # Collapse the tail into a single "other features" bar.
    if len(vals_d) > max_display:
        keep = max_display - 1
        rest_sum = vals_d[keep:].sum()
        vals_d = np.append(vals_d[:keep], rest_sum)
        names_d = names_d[:keep] + [f'{len(order) - keep} other features']
 
    # Largest at the top: reverse so y=0 (bottom) is the smallest contributor,
    # then accumulate from the base value upward.
    vals_by_y = vals_d[::-1]
    names_by_y = names_d[::-1]
    starts = base_value + np.concatenate([[0.0], np.cumsum(vals_by_y)[:-1]])
    y = np.arange(len(vals_by_y))
 
    for yi, start, v in zip(y, starts, vals_by_y):
        ax.barh(yi, v, left=start,
                color=(pos_color if v >= 0 else neg_color),
                edgecolor='white')
        if annotate and value_fmt is not None:
            ax.text(start + v, yi, ' ' + value_fmt.format(v),
                    va='center',
                    ha=('left' if v >= 0 else 'right'),
                    fontsize=10)
 
    final = base_value + vals_d.sum()
    ax.axvline(base_value, color='gray', ls='--', lw=0.8, label='E[f(X)]')
    ax.axvline(final, color='black', ls=':', lw=0.8, label='f(x)')
 
    ax.set_yticks(y)
    ax.set_yticklabels(names_by_y)
    ax.set_xlabel('Model output')
    ax.set_title(title)
    return ax