import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import t

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from math import pi

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15

def save_publication_figure(filename, dpi=300):
    plt.savefig(f"{filename}.png", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{filename}.pdf", format='pdf', bbox_inches='tight')
    print(f"Salvo {filename} (.png e .pdf)")


def create_radar_chart(df, selected_models=['LogisticRegression', 'GaussianNB', 'MLP', 'RandomForest']):
    categories_en = ['Accuracy', 'F2-Score', 'Precision', 'Recall', 'AUC']
    categories_pt = ['Acurácia', 'F2-Score', 'Precisão', 'Recall', 'AUC']
    
    df_filtered = df[df['Model'].isin(selected_models)].copy()
    
    df_radar = df_filtered.groupby('Model')[categories_en].mean().reset_index()

    model_map = {
        'GaussianNB': 'Naive Bayes',
        'RandomForest': 'Random Forest',
        'DecisionTree-pruned': 'Árvore de Decisão',
        'MLP': 'Rede Neural',
        'LogisticRegression': 'Regressão Logística'
    }
    df_radar['Model'] = df_radar['Model'].map(model_map).fillna(df_radar['Model'])

    N = len(categories_en)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(6, 6)) 
    ax = plt.subplot(111, polar=True)
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories_pt, size=15, fontweight='bold')
    
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], 
               color="grey", size=12)
    plt.ylim(0, 1.15) 
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray', zorder=1)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='gray', zorder=1)

    colors = sns.color_palette("colorblind", len(selected_models))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

    for i, model_name in enumerate(df_radar['Model']):
        row = df_radar[df_radar['Model'] == model_name]
        
        values = row[categories_en].values.flatten().tolist()
        values += values[:1]
        
        color = colors[i]
        marker = markers[i % len(markers)]
        
        ax.plot(angles, values, 
                linewidth=2.5,
                linestyle='solid', 
                label=model_name, 
                color=color,
                marker=marker,
                markersize=8,
                zorder=i+2)

    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.05), 
               title='Modelo', frameon=False)
    
    plt.title("Comparação de Desempenho do Modelo", y=1.12, fontweight='bold')
    
    save_publication_figure("Figura_3_Grafico_Radar_Melhorado")
    plt.close()


def create_combined_metric_plot(df):
    metrics = ['Accuracy', 'F2-Score', 'Precision', 'Recall', 'AUC']
    
    if not all(col in df.columns for col in metrics):
        print(f"Erro: Dataframe não possui uma das colunas: {metrics}")
        return

    print("Gerando comparação combinada de métricas...")

    df_long = df.melt(
        id_vars=['Model'], 
        value_vars=metrics, 
        var_name='Métrica', 
        value_name='Pontuação'
    )

    model_map = {
        'GaussianNB': 'NB',
        'RandomForest': 'RF',
        'DecisionTree-pruned': 'Árvore',
        'MLP': 'Neural',
        'LogisticRegression': 'RL'
    }
    df_long['Model'] = df_long['Model'].map(model_map).fillna(df_long['Model'])

    metric_map = {
        'Accuracy': 'Acurácia',
        'F2-Score': 'F2-Score',
        'Precision': 'Precisão',
        'Recall': 'Recall',
        'AUC': 'AUC'
    }
    df_long['Métrica'] = df_long['Métrica'].map(metric_map)

    sns.set_style("ticks")
    
    g = sns.catplot(
        data=df_long, 
        x='Model', 
        y='Pontuação', 
        col='Métrica', 
        kind='box',
        palette='colorblind',
        height=3,
        aspect=0.8,
        sharey=False,
        linewidth=1.5,
        fliersize=3
    )

    g.figure.subplots_adjust(wspace=0.4, hspace=0.5)

    g.set_titles("{col_name}", fontweight='bold', size=16) 
    g.set_axis_labels("", "Pontuação")
    
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')
            label.set_size(12) 

    for ax in g.axes.flat:
        ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray')

    save_publication_figure("Figura_1_Comparacao_Metricas_Modelo")
    plt.close()


def create_pr_scatter(df):
    print("\nGerando gráfico de dispersão Precisão-Recall...")

    summary = df.groupby('Model').agg(
        recall_mean=('Recall', 'mean'),
        recall_std=('Recall', 'std'),
        recall_count=('Recall', 'count'),
        precision_mean=('Precision', 'mean'),
        precision_std=('Precision', 'std'),
        precision_count=('Precision', 'count')
    )

    model_map = {
        'GaussianNB': 'Naive Bayes',
        'RandomForest': 'Random Forest',
        'DecisionTree-pruned': 'Árvore de Decisão',
        'MLP': 'Rede Neural',
        'LogisticRegression': 'Regressão Logística'
    }
    new_index = summary.index.map(model_map)
    
    filled_index = np.where(pd.isna(new_index), summary.index, new_index)
    summary.index = filled_index


    def calc_ci_err(row, metric):
        n = row[f'{metric}_count']
        if n < 2: return 0
        t_crit = t.ppf(0.975, n - 1)
        return t_crit * (row[f'{metric}_std'] / np.sqrt(n))

    summary['recall_err'] = summary.apply(calc_ci_err, axis=1, metric='recall')
    summary['precision_err'] = summary.apply(calc_ci_err, axis=1, metric='precision')

    sns.set_style("white")
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    if len(summary) > len(markers):
        markers = ['o'] * len(summary)
        
    colors = sns.color_palette('colorblind', n_colors=len(summary))

    for i, model in enumerate(summary.index):
        row = summary.loc[model]
        
        ax.errorbar(
            x=row['recall_mean'],
            y=row['precision_mean'],
            xerr=row['recall_err'],
            yerr=row['precision_err'],
            fmt='none',
            ecolor='gray',
            elinewidth=1.5,
            capsize=4,
            alpha=0.6,
            zorder=1
        )
        
        ax.scatter(
            x=row['recall_mean'],
            y=row['precision_mean'],
            label=model,
            color=colors[i],
            marker=markers[i % len(markers)],
            s=100,
            edgecolor='white',
            linewidth=0.5,
            zorder=2
        )

    lims = [0, 1]
    ax.plot(lims, lims, 'k--', alpha=0.2, zorder=0, label='Balanceado (y=x)')
    
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precisão', fontweight='bold')
    
    all_vals = np.concatenate([summary['recall_mean'], summary['precision_mean']])
    min_v = max(0, all_vals.min() - 0.1)
    max_v = min(1, all_vals.max() + 0.1)
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    
    sns.despine(trim=True)
    
    plt.legend(title='Modelo', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    save_publication_figure("Figura_2_Precisao_vs_Revocacao")
    plt.close()

def main():
    filename = 'model_results.csv'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        
    try:
        df = pd.read_csv(filename)
        create_combined_metric_plot(df)
        create_pr_scatter(df)
        
        print("\nGerando Gráfico Radar Melhorado...")
        create_radar_chart(df) 
        
        print("\nPronto. Verifique os arquivos Figura_1..., Figura_2... e Figura_3...")
    except FileNotFoundError:
        print(f"Erro: {filename} não encontrado.")

if __name__ == "__main__":
    main()