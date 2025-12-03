import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
import numpy as np

# Gera gráficos para o dataset DiaBD e realiza análise exploratória de dados.

def analyze_health_data():
    try:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['font.size'] = 10
        # ---------------------------------------------
        try:
            df = pd.read_csv('diabd_dataset.csv')
            print("Arquivo 'diabd_dataset.csv' carregado com sucesso.")
        except FileNotFoundError:
            print("Erro: Arquivo 'diabd_dataset.csv' não encontrado.")
            return

        sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)
        plt.rcParams['text.color'] = '#333333'

        COLOR_MAP = {
            'No': '#377eb8',  # Azul sólido
            'Yes': '#e41a1c'  # Vermelho distinto
        }


        plt.figure(figsize=(8, 5))
        sns.histplot(df['age'].dropna(), kde=True, bins=12, 
                     color='#2c5f8e', alpha=0.8, edgecolor='white', linewidth=0.5)
        plt.title('Distribuição de Idade', fontweight='bold', pad=15)
        plt.xlabel('Idade (anos)', fontweight='bold')
        plt.ylabel('Frequência', fontweight='bold')
        plt.tight_layout()
        plt.savefig('age_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()


        plt.figure(figsize=(8, 5))
        bin_width = 1.5
        g_min, g_max = df['glucose'].min(), df['glucose'].max()
        
        bins_lower = np.arange(7.8, g_min - bin_width, -bin_width)
        bins_upper = np.arange(7.8, g_max + bin_width, bin_width)
        custom_bins = np.sort(np.unique(np.concatenate([bins_lower, bins_upper])))

        sns.histplot(df['glucose'].dropna(), kde=True, bins=custom_bins, 
                     color='#388e3c', alpha=0.8, edgecolor='white', linewidth=0.5)
        
        plt.axvline(x=7.8, color='#e41a1c', linestyle='--', linewidth=3, label='Limiar (7.8)')
        plt.legend(fontsize=14, frameon=True, facecolor='white', framealpha=0.9)

        plt.title('Distribuição de Glicose', fontweight='bold', pad=15)
        plt.xlabel('Glicose (mmol/L)', fontweight='bold')
        plt.ylabel('Frequência', fontweight='bold')
        plt.tight_layout()
        plt.savefig('glucose_histogram_aligned.png', dpi=300, bbox_inches='tight')
        plt.close()


        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        def create_contrasted_pie(ax, data, gender_val, title_text):
            gender_df = data[data['gender'] == gender_val]
            counts = gender_df.groupby('diabetic').size().sort_index()
            pie_colors = [COLOR_MAP.get(x, '#333333') for x in counts.index]
            labels_pt = [{'No': 'Não Diabético', 'Yes': 'Diabético'}.get(x) for x in counts.index]

            wedges, texts, autotexts = ax.pie(
                counts, autopct='%1.1f%%', startangle=60, colors=pie_colors,
                wedgeprops={'edgecolor': 'black', 'linewidth': 2.5}, pctdistance=0.6
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
                autotext.set_fontsize(16)
            
            ax.set_title(title_text, fontweight='bold', fontsize=18, pad=10)
            ax.legend(wedges, labels_pt, loc="lower center", 
                     bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=2, fontsize=14)

        try:
            create_contrasted_pie(axs[0], df, 'Female', 'Gênero Feminino')
            create_contrasted_pie(axs[1], df, 'Male', 'Gênero Masculino')
        except Exception as e:
            print(f"Erro ao gerar pizzas: {e}")

        fig.suptitle('Prevalência de Diabetes por Gênero', fontsize=22, fontweight='bold', y=0.95)
        plt.subplots_adjust(wspace=0.5)
        plt.savefig('diabetics_by_gender_side_by_side.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Dados em análise: {len(df)} registros.\n")
        
        all_cols = [
            "age", "gender", "pulse_rate", "systolic_bp", "diastolic_bp", 
            "glucose", "height", "weight", "bmi", "family_diabetes", 
            "hypertensive", "family_hypertension", "cardiovascular_disease", 
            "stroke", "diabetic"
        ]
        pt_labels = {
            "age": "Idade", "gender": "Gênero", "pulse_rate": "Freq. Cardíaca",
            "systolic_bp": "Pressão Sistólica", "diastolic_bp": "Pressão Diastólica",
            "glucose": "Glicose", "height": "Altura", "weight": "Peso", "bmi": "IMC",
            "family_diabetes": "Família c/ Diabetes", "hypertensive": "Hipertenso(a)",
            "family_hypertension": "Família c/ Hipertensão",
            "cardiovascular_disease": "Doença Cardiovascular", "stroke": "AVC",
            "diabetic": "Diabético(a)"
        }
        
        cat_cols = [
            "gender", "family_diabetes", "hypertensive", 
            "family_hypertension", "cardiovascular_disease", 
            "stroke", "diabetic"
        ]

        cols_to_use = [col for col in all_cols if col in df.columns]
        cat_cols_to_use = [col for col in cat_cols if col in cols_to_use]
        
        if not cols_to_use:
            print("Erro: Nenhuma das colunas esperadas foi encontrada no DataFrame.")
            return

        df_corr = df[cols_to_use].copy()
        
        for col in cat_cols_to_use:
            df_corr[col] = pd.factorize(df_corr[col])[0]
            
        corr_matrix = df_corr.corr()
        
        pt_labels_to_use = {k: v for k, v in pt_labels.items() if k in cols_to_use}
        corr_matrix_pt = corr_matrix.rename(index=pt_labels_to_use, columns=pt_labels_to_use)

        plt.figure(figsize=(15, 12))
        
        mask = np.triu(np.ones_like(corr_matrix_pt, dtype=bool))

        heatmap = sns.heatmap(
            corr_matrix_pt,
            mask=mask,               
            annot=True,             
            fmt=".2f",               
            cmap="vlag",             
            linewidths=.5,           
            square=True,             
            cbar_kws={"shrink": .8, "label": "Coeficiente de Correlação"}, 
            annot_kws={"size": 9}   
        )
        
        plt.title("Matriz de Correlação Dataset DiaBD", fontsize=20, fontweight='bold', pad=20) 
        plt.xticks(rotation=45, ha='right', fontsize=15) 
        plt.yticks(rotation=0, fontsize=15)           
        heatmap.figure.axes[-1].yaxis.label.set_size(14) 
        
        plt.tight_layout(pad=1.5)
        
        output_filename = "correlation_heatmap_pt.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print("Análise concluída e gráficos salvos com sucesso.")
    except Exception as e:
        print(f"Um erro inesperado ocorreu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    
    analyze_health_data()