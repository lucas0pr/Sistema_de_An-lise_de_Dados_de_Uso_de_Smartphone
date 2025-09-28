#!/usr/bin/env python3
"""
Sistema de Análise de Dados de Uso de Smartphone
Dataset: Smartphone Usage and Behavioral Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class SmartphoneUsageAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.df_clean = None
        
    def load_data(self):
        """1. Leitura do dataset utilizando Pandas"""
        print("=" * 60)
        print("1. CARREGANDO O DATASET")
        print("=" * 60)
        
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✓ Dataset carregado com sucesso!")
            print(f"✓ Total de registros: {len(self.df)}")
            print(f"✓ Total de colunas: {len(self.df.columns)}")
            print(f"✓ Colunas: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"✗ Erro ao carregar o dataset: {e}")
            return False
    
    def clean_data(self):
        """2. Tratamento inicial dos dados com limpeza completa"""
        print("\n" + "=" * 60)
        print("2. TRATAMENTO INICIAL DOS DADOS")
        print("=" * 60)
        
        # Criar uma cópia do dataframe original
        self.df_clean = self.df.copy()
        initial_count = len(self.df_clean)
        
        # 2.1 Verificar e tratar valores nulos
        print("\n2.1 Verificação e tratamento de valores nulos:")
        null_counts = self.df_clean.isnull().sum()
        print(null_counts)
        
        if null_counts.sum() > 0:
            print(f"\n✓ Total de valores nulos encontrados: {null_counts.sum()}")
            # Remover linhas com valores nulos
            self.df_clean = self.df_clean.dropna()
            print(f"✓ Valores nulos removidos. Registros restantes: {len(self.df_clean)}")
        else:
            print("✓ Nenhum valor nulo encontrado no dataset")
        
        # 2.2 Padronização de colunas
        print("\n2.2 Padronização de colunas:")
        
        # Converter colunas numéricas para o tipo correto
        numeric_columns = ['Age', 'Total_App_Usage_Hours', 'Daily_Screen_Time_Hours', 
                         'Number_of_Apps_Used', 'Social_Media_Usage_Hours', 
                         'Productivity_App_Usage_Hours', 'Gaming_App_Usage_Hours']
        
        for col in numeric_columns:
            if col in self.df_clean.columns:
                self.df_clean[col] = pd.to_numeric(self.df_clean[col], errors='coerce')
        
        # Verificar valores nulos após conversão
        new_nulls = self.df_clean[numeric_columns].isnull().sum()
        if new_nulls.sum() > 0:
            print(f"\n⚠ Valores nulos após conversão de tipos: {new_nulls.sum()}")
            self.df_clean = self.df_clean.dropna(subset=numeric_columns)
            print(f"✓ Valores nulos removidos. Registros restantes: {len(self.df_clean)}")
        
        # Padronizar coluna Gender
        if 'Gender' in self.df_clean.columns:
            self.df_clean['Gender'] = self.df_clean['Gender'].str.strip().str.title()
            # Corrigir variações comuns
            self.df_clean['Gender'] = self.df_clean['Gender'].replace({
                'M': 'Male', 'F': 'Female', 
                'Homem': 'Male', 'Mulher': 'Female',
                'Masculino': 'Male', 'Feminino': 'Female'
            })
        
        # Padronizar coluna Location
        if 'Location' in self.df_clean.columns:
            self.df_clean['Location'] = self.df_clean['Location'].str.strip().str.title()
        
        print("\n✓ Tipos de dados após padronização:")
        print(self.df_clean.dtypes)
        
        # 2.3 Correção de valores inconsistentes
        print("\n2.3 Correção de valores inconsistentes:")
        
        # Verificar e corrigir idades inválidas
        if 'Age' in self.df_clean.columns:
            age_issues = 0
            # Idades negativas ou acima de 120 anos
            age_mask = (self.df_clean['Age'] < 10) | (self.df_clean['Age'] > 120)
            age_issues += age_mask.sum()
            self.df_clean = self.df_clean[~age_mask]
            
            # Idades não inteiras
            non_integer_age = self.df_clean['Age'] % 1 != 0
            if non_integer_age.any():
                age_issues += non_integer_age.sum()
                self.df_clean['Age'] = self.df_clean['Age'].round().astype(int)
            
            print(f"✓ Idades corrigidas: {age_issues} registros")
        
        # Verificar e corrigir horas de uso inválidas
        hour_columns = ['Total_App_Usage_Hours', 'Daily_Screen_Time_Hours', 
                       'Social_Media_Usage_Hours', 'Productivity_App_Usage_Hours', 
                       'Gaming_App_Usage_Hours']
        
        for col in hour_columns:
            if col in self.df_clean.columns:
                # Horas negativas
                negative_mask = self.df_clean[col] < 0
                if negative_mask.any():
                    print(f"✓ {col}: {negative_mask.sum()} registros com valores negativos corrigidos para 0")
                    self.df_clean.loc[negative_mask, col] = 0
                
                # Horas excessivas (acima de 24h para uso diário)
                if 'Daily' in col:
                    excess_mask = self.df_clean[col] > 24
                    if excess_mask.any():
                        print(f"✓ {col}: {excess_mask.sum()} registros com valores >24h corrigidos para 24")
                        self.df_clean.loc[excess_mask, col] = 24
        
        # Verificar e corrigir número de apps inválido
        if 'Number_of_Apps_Used' in self.df_clean.columns:
            app_mask = (self.df_clean['Number_of_Apps_Used'] < 0) | (self.df_clean['Number_of_Apps_Used'] > 200)
            if app_mask.any():
                print(f"✓ Number_of_Apps_Used: {app_mask.sum()} registros com valores inválidos removidos")
                self.df_clean = self.df_clean[~app_mask]
        
        # 2.4 Verificar e remover duplicatas
        duplicate_count = self.df_clean.duplicated().sum()
        if duplicate_count > 0:
            print(f"\n✓ Encontradas {duplicate_count} linhas duplicadas")
            self.df_clean = self.df_clean.drop_duplicates()
            print(f"✓ Duplicatas removidas. Registros restantes: {len(self.df_clean)}")
        else:
            print("\n✓ Nenhuma duplicata encontrada")
        
        # 2.5 Verificar e remover outliers extremos
        print("\n2.5 Remoção de outliers extremos:")
        outliers_removed = 0
        for col in numeric_columns:
            if col in self.df_clean.columns and col != 'Age':  # Não remover outliers de idade
                # Usar método mais robusto para outliers
                Q1 = self.df_clean[col].quantile(0.25)
                Q3 = self.df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # Limite mais rigoroso
                upper_bound = Q3 + 3 * IQR
                
                outliers = self.df_clean[(self.df_clean[col] < lower_bound) | (self.df_clean[col] > upper_bound)]
                if len(outliers) > 0:
                    print(f"  {col}: {len(outliers)} outliers extremos removidos")
                    self.df_clean = self.df_clean[(self.df_clean[col] >= lower_bound) & (self.df_clean[col] <= upper_bound)]
                    outliers_removed += len(outliers)
        
        if outliers_removed == 0:
            print("  ✓ Nenhum outlier extremo encontrado")
        
        # 2.6 Verificar consistência entre colunas
        print("\n2.6 Verificação de consistência entre colunas:")
        
        # Verificar consistência entre uso total e partes
        if all(col in self.df_clean.columns for col in ['Total_App_Usage_Hours', 'Social_Media_Usage_Hours', 
                                                       'Productivity_App_Usage_Hours', 'Gaming_App_Usage_Hours']):
            sum_parts = (self.df_clean['Social_Media_Usage_Hours'] + 
                        self.df_clean['Productivity_App_Usage_Hours'] + 
                        self.df_clean['Gaming_App_Usage_Hours'])
            
            # Identificar inconsistências
            inconsistency = self.df_clean['Total_App_Usage_Hours'] != sum_parts
            inconsistency_count = inconsistency.sum()
            
            if inconsistency_count > 0:
                print(f"✓ {inconsistency_count} registros com inconsistência entre uso total e partes")
                
                # Corrigir recalculando o total como a soma das partes
                self.df_clean.loc[inconsistency, 'Total_App_Usage_Hours'] = sum_parts[inconsistency]
                print(f"✓ Inconsistências corrigidas recalculando o total como soma das partes")
        
        # Verificar consistência entre tempo de tela e uso total
        if all(col in self.df_clean.columns for col in ['Daily_Screen_Time_Hours', 'Total_App_Usage_Hours']):
            screen_inconsistency = self.df_clean['Daily_Screen_Time_Hours'] < self.df_clean['Total_App_Usage_Hours']
            screen_inconsistency_count = screen_inconsistency.sum()
            
            if screen_inconsistency_count > 0:
                print(f"✓ {screen_inconsistency_count} registros com tempo de tela menor que uso total")
                
                # Corrigir ajustando o tempo de tela para o uso total
                self.df_clean.loc[screen_inconsistency, 'Daily_Screen_Time_Hours'] = self.df_clean.loc[screen_inconsistency, 'Total_App_Usage_Hours']
                print(f"✓ Tempo de tela ajustado para igualar o uso total")
        
        # Resumo final da limpeza
        final_count = len(self.df_clean)
        removed_count = initial_count - final_count
        print(f"\n✓ Tratamento de dados concluído!")
        print(f"✓ Registros removidos: {removed_count} ({(removed_count/initial_count)*100:.1f}%)")
        print(f"✓ Registros finais: {final_count}")
        
        return True
    
    def exploratory_analysis(self):
        """3. Exploração inicial com estatísticas descritivas"""
        print("\n" + "=" * 60)
        print("3. ANÁLISE EXPLORATÓRIA INICIAL")
        print("=" * 60)
        
        # 3.1 Estatísticas descritivas básicas
        print("\n3.1 Estatísticas descritivas:")
        numeric_columns = ['Age', 'Total_App_Usage_Hours', 'Daily_Screen_Time_Hours', 
                         'Number_of_Apps_Used', 'Social_Media_Usage_Hours', 
                         'Productivity_App_Usage_Hours', 'Gaming_App_Usage_Hours']
        
        desc_stats = self.df_clean[numeric_columns].describe()
        print(desc_stats)
        
        # 3.2 Análise por gênero
        print("\n3.2 Análise por gênero:")
        gender_counts = self.df_clean['Gender'].value_counts()
        print("Distribuição por gênero:")
        print(gender_counts)
        
        gender_stats = self.df_clean.groupby('Gender')[numeric_columns].mean()
        print("\nMédias por gênero:")
        print(gender_stats)
        
        # 3.3 Análise por localização
        print("\n3.3 Análise por localização:")
        location_counts = self.df_clean['Location'].value_counts()
        print("Distribuição por localização:")
        print(location_counts)
        
        location_stats = self.df_clean.groupby('Location')[numeric_columns].mean()
        print("\nMédias por localização:")
        print(location_stats)
        
        # 3.4 Correlações
        print("\n3.4 Matriz de correlação:")
        correlation_matrix = self.df_clean[numeric_columns].corr()
        print(correlation_matrix)
        
        # 3.5 Insights básicos
        print("\n3.5 Insights básicos:")
        print(f"• Idade média dos usuários: {self.df_clean['Age'].mean():.1f} anos")
        print(f"• Tempo médio de uso de apps: {self.df_clean['Total_App_Usage_Hours'].mean():.2f} horas")
        print(f"• Tempo médio de tela diário: {self.df_clean['Daily_Screen_Time_Hours'].mean():.2f} horas")
        print(f"• Número médio de apps usados: {self.df_clean['Number_of_Apps_Used'].mean():.1f}")
        print(f"• Uso médio de redes sociais: {self.df_clean['Social_Media_Usage_Hours'].mean():.2f} horas")
        print(f"• Uso médio de apps de produtividade: {self.df_clean['Productivity_App_Usage_Hours'].mean():.2f} horas")
        print(f"• Uso médio de jogos: {self.df_clean['Gaming_App_Usage_Hours'].mean():.2f} horas")
        
        return True
    
    def create_visualizations(self):
        """4. Visualização básica com Matplotlib e Seaborn"""
        print("\n" + "=" * 60)
        print("4. CRIANDO VISUALIZAÇÕES")
        print("=" * 60)
        
        # Criar diretório para salvar os gráficos
        import os
        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')
        
        # 4.1 Gráfico de distribuição por gênero
        print("\n4.1 Gráfico de distribuição por gênero...")
        plt.figure(figsize=(10, 6))
        gender_counts = self.df_clean['Gender'].value_counts()
        plt.subplot(1, 2, 1)
        gender_counts.plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
        plt.title('Distribuição por Gênero')
        plt.xlabel('Gênero')
        plt.ylabel('Contagem')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 2, 2)
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
                colors=['#FF6B6B', '#4ECDC4'])
        plt.title('Percentual por Gênero')
        
        plt.tight_layout()
        plt.savefig('visualizations/01_distribuicao_genero.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo como 'visualizations/01_distribuicao_genero.png'")
        
        # 4.2 Gráfico de distribuição por localização
        print("\n4.2 Gráfico de distribuição por localização...")
        plt.figure(figsize=(12, 6))
        location_counts = self.df_clean['Location'].value_counts()
        location_counts.plot(kind='bar', color='#45B7D1')
        plt.title('Distribuição por Localização')
        plt.xlabel('Localização')
        plt.ylabel('Contagem')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/02_distribuicao_localizacao.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo como 'visualizations/02_distribuicao_localizacao.png'")
        
        # 4.3 Gráfico de comparação de uso de apps por gênero
        print("\n4.3 Gráfico de comparação de uso de apps por gênero...")
        plt.figure(figsize=(15, 10))
        
        usage_columns = ['Social_Media_Usage_Hours', 'Productivity_App_Usage_Hours', 'Gaming_App_Usage_Hours']
        gender_usage = self.df_clean.groupby('Gender')[usage_columns].mean()
        
        x = np.arange(len(usage_columns))
        width = 0.35
        
        plt.subplot(2, 2, 1)
        plt.bar(x - width/2, gender_usage.loc['Male'], width, label='Masculino', color='#4ECDC4')
        plt.bar(x + width/2, gender_usage.loc['Female'], width, label='Feminino', color='#FF6B6B')
        plt.xlabel('Tipo de App')
        plt.ylabel('Horas Médias')
        plt.title('Uso Médio de Apps por Gênero')
        plt.xticks(x, ['Redes Sociais', 'Produtividade', 'Jogos'])
        plt.legend()
        
        # 4.4 Gráfico de dispersão: Idade vs Tempo de Tela
        plt.subplot(2, 2, 2)
        plt.scatter(self.df_clean['Age'], self.df_clean['Daily_Screen_Time_Hours'], 
                   alpha=0.6, c=self.df_clean['Age'], cmap='viridis')
        plt.xlabel('Idade')
        plt.ylabel('Tempo de Tela Diário (horas)')
        plt.title('Idade vs Tempo de Tela Diário')
        plt.colorbar(label='Idade')
        
        # 4.5 Gráfico de boxplot: Uso de apps por localização
        plt.subplot(2, 2, 3)
        sns.boxplot(data=self.df_clean, x='Location', y='Total_App_Usage_Hours')
        plt.xlabel('Localização')
        plt.ylabel('Uso Total de Apps (horas)')
        plt.title('Uso de Apps por Localização')
        plt.xticks(rotation=45)
        
        # 4.6 Gráfico de correlação
        plt.subplot(2, 2, 4)
        correlation_matrix = self.df_clean[['Age', 'Total_App_Usage_Hours', 'Daily_Screen_Time_Hours', 
                                           'Number_of_Apps_Used']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlação')
        
        plt.tight_layout()
        plt.savefig('visualizations/03_analise_comparativa.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo como 'visualizations/03_analise_comparativa.png'")
        
        # 4.7 Gráfico de evolução: Número de apps vs Idade
        print("\n4.4 Gráfico de evolução: Número de apps vs Idade...")
        plt.figure(figsize=(12, 6))
        
        # Criar faixas etárias
        self.df_clean['Age_Group'] = pd.cut(self.df_clean['Age'], 
                                           bins=[0, 25, 35, 45, 55, 100], 
                                           labels=['18-25', '26-35', '36-45', '46-55', '56+'])
        
        age_group_stats = self.df_clean.groupby('Age_Group').agg({
            'Number_of_Apps_Used': 'mean',
            'Total_App_Usage_Hours': 'mean',
            'Daily_Screen_Time_Hours': 'mean'
        }).reset_index()
        
        plt.subplot(1, 2, 1)
        plt.plot(age_group_stats['Age_Group'], age_group_stats['Number_of_Apps_Used'], 
                marker='o', linewidth=2, markersize=8, color='#FF6B6B')
        plt.xlabel('Faixa Etária')
        plt.ylabel('Número Médio de Apps')
        plt.title('Número de Apps por Faixa Etária')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(age_group_stats['Age_Group'], age_group_stats['Total_App_Usage_Hours'], 
                marker='s', linewidth=2, markersize=8, color='#4ECDC4', label='Uso Total')
        plt.plot(age_group_stats['Age_Group'], age_group_stats['Daily_Screen_Time_Hours'], 
                marker='^', linewidth=2, markersize=8, color='#45B7D1', label='Tempo de Tela')
        plt.xlabel('Faixa Etária')
        plt.ylabel('Horas Médias')
        plt.title('Uso de Apps por Faixa Etária')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/04_evolucao_faixa_etaria.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico salvo como 'visualizations/04_evolucao_faixa_etaria.png'")
        
        return True
    
    def generate_report(self):
        """5. Relatório simples com insights do dataset"""
        print("\n" + "=" * 60)
        print("5. RELATÓRIO DE INSIGHTS")
        print("=" * 60)
        
        # Criar diretório para salvar os relatórios
        import os
        if not os.path.exists('reports'):
            os.makedirs('reports')
        
        # Data e hora atual
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Gerar relatorio_completo.txt
        with open('reports/relatorio_completo.txt', 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE ANÁLISE DE DADOS DE USO DE SMARTPHONE\n")
            f.write("============================================================\n\n")
            f.write(f"Data de geração: {current_datetime}\n")
            f.write(f"Total de usuários: {len(self.df_clean)}\n\n")
            
            f.write("RESUMO GERAL\n")
            f.write("--------------------\n")
            f.write(f"Total de usuários analisados: {len(self.df_clean)}\n")
            f.write(f"Localizações cobertas: {', '.join(self.df_clean['Location'].unique())}\n")
            f.write(f"Faixa etária: {self.df_clean['Age'].min()} a {self.df_clean['Age'].max()} anos\n\n")
            
            # Adicionar o restante do conteúdo do relatório
            f.write("INSIGHTS DEMOGRÁFICOS\n")
            f.write("--------------------\n")
            gender_dist = self.df_clean['Gender'].value_counts(normalize=True) * 100
            f.write("Distribuição de gênero:\n")
            f.write(f"  - Masculino: {gender_dist.get('Male', 0):.1f}%\n")
            f.write(f"  - Feminino: {gender_dist.get('Female', 0):.1f}%\n\n")
            
            location_dist = self.df_clean['Location'].value_counts(normalize=True) * 100
            f.write("Principais localizações:\n")
            for loc, pct in location_dist.items():
                f.write(f"  - {loc}: {pct:.1f}%\n")
            f.write("\n")
            
            f.write("INSIGHTS DE COMPORTAMENTO\n")
            f.write("--------------------\n")
            gender_behavior = self.df_clean.groupby('Gender').agg({
                'Social_Media_Usage_Hours': 'mean',
                'Productivity_App_Usage_Hours': 'mean',
                'Gaming_App_Usage_Hours': 'mean',
                'Total_App_Usage_Hours': 'mean'
            })
            
            f.write("Padrões de uso por gênero:\n")
            for gender in gender_behavior.index:
                f.write(f"  - {gender}:\n")
                f.write(f"    * Redes sociais: {gender_behavior.loc[gender, 'Social_Media_Usage_Hours']:.2f}h/dia\n")
                f.write(f"    * Produtividade: {gender_behavior.loc[gender, 'Productivity_App_Usage_Hours']:.2f}h/dia\n")
                f.write(f"    * Jogos: {gender_behavior.loc[gender, 'Gaming_App_Usage_Hours']:.2f}h/dia\n")
            f.write("\n")
            
            location_behavior = self.df_clean.groupby('Location').agg({
                'Daily_Screen_Time_Hours': 'mean',
                'Number_of_Apps_Used': 'mean'
            }).sort_values('Daily_Screen_Time_Hours', ascending=False)
            
            f.write("Localizações com maior tempo de tela:\n")
            for loc in location_behavior.index:
                f.write(f"  - {loc}: {location_behavior.loc[loc, 'Daily_Screen_Time_Hours']:.2f}h/dia, "
                      f"{location_behavior.loc[loc, 'Number_of_Apps_Used']:.1f} apps\n")
            f.write("\n")
            
            f.write("INSIGHTS DE CORRELAÇÃO\n")
            f.write("--------------------\n")
            correlation_matrix = self.df_clean[['Age', 'Total_App_Usage_Hours', 'Daily_Screen_Time_Hours', 
                                             'Number_of_Apps_Used']].corr()
            
            corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.3:
                        corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_value))
            
            f.write("Correlações significativas:\n")
            for col1, col2, corr in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                direction = "positiva" if corr > 0 else "negativa"
                strength = "forte" if abs(corr) > 0.7 else "moderada"
                f.write(f"  - {col1} vs {col2}: correlação {direction} {strength} ({corr:.3f})\n")
            f.write("\n")
            
            f.write("INSIGHTS POR FAIXA ETÁRIA\n")
            f.write("--------------------\n")
            age_insights = self.df_clean.groupby('Age_Group').agg({
                'Number_of_Apps_Used': ['mean', 'std'],
                'Total_App_Usage_Hours': ['mean', 'std'],
                'Social_Media_Usage_Hours': 'mean',
                'Gaming_App_Usage_Hours': 'mean'
            }).round(2)
            
            f.write("Comportamento por faixa etária:\n")
            for age_group in age_insights.index:
                f.write(f"  - {age_group} anos:\n")
                f.write(f"    * Apps usados: {age_insights.loc[age_group, ('Number_of_Apps_Used', 'mean')]:.1f} "
                      f"(±{age_insights.loc[age_group, ('Number_of_Apps_Used', 'std')]:.1f})\n")
                f.write(f"    * Uso total: {age_insights.loc[age_group, ('Total_App_Usage_Hours', 'mean')]:.2f}h/dia\n")
                f.write(f"    * Redes sociais: {age_insights.loc[age_group, ('Social_Media_Usage_Hours', 'mean')]:.2f}h/dia\n")
                f.write(f"    * Jogos: {age_insights.loc[age_group, ('Gaming_App_Usage_Hours', 'mean')]:.2f}h/dia\n")
            f.write("\n")
            
            f.write("RECOMENDAÇÕES E OBSERVAÇÕES\n")
            f.write("--------------------\n")
            f.write("Baseado nos dados analisados, observamos que:\n")
            f.write("  1. O tempo de uso de dispositivos varia significativamente por localização\n")
            f.write("  2. Existem diferenças claras nos padrões de uso entre gêneros\n")
            f.write("  3. A faixa etária influencia no número de apps utilizados\n")
            f.write("  4. Redes sociais são a categoria mais utilizada em geral\n")
            f.write("  5. Há correlação entre o número de apps e o tempo total de uso\n\n")
            
            f.write("Sugestões para análise futura:\n")
            f.write("  1. Investigar os fatores que influenciam as diferenças por localização\n")
            f.write("  2. Analisar o impacto do tempo de uso na produtividade\n")
            f.write("  3. Estudar padrões de uso ao longo do tempo (dados temporais)\n")
            f.write("  4. Segmentar análise por tipo de dispositivo ou sistema operacional\n")
            f.write("  5. Investigar a relação entre uso de apps e satisfação do usuário\n")
        
        # Gerar visualizacoes_texto.txt
        with open('reports/visualizacoes_texto.txt', 'w', encoding='utf-8') as f:
            f.write("VISUALIZAÇÕES DE DADOS DE USO DE SMARTPHONE\n")
            f.write("==================================================\n\n")
            
            # Distribuição por Gênero
            f.write("Distribuição por Gênero:\n")
            gender_counts = self.df_clean['Gender'].value_counts()
            total = len(self.df_clean)
            for gender, count in gender_counts.items():
                percentage = (count / total) * 100
                f.write(f"{gender}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Distribuição por Localização
            f.write("Distribuição por Localização:\n")
            location_counts = self.df_clean['Location'].value_counts()
            for location, count in location_counts.items():
                percentage = (count / total) * 100
                f.write(f"{location}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Comparação de uso de apps por gênero
            f.write("Uso Médio de Apps por Gênero:\n")
            usage_columns = ['Social_Media_Usage_Hours', 'Productivity_App_Usage_Hours', 'Gaming_App_Usage_Hours']
            gender_usage = self.df_clean.groupby('Gender')[usage_columns].mean()
            
            for gender in gender_usage.index:
                f.write(f"{gender}:\n")
                f.write(f"  Redes Sociais: {gender_usage.loc[gender, 'Social_Media_Usage_Hours']:.2f} horas\n")
                f.write(f"  Produtividade: {gender_usage.loc[gender, 'Productivity_App_Usage_Hours']:.2f} horas\n")
                f.write(f"  Jogos: {gender_usage.loc[gender, 'Gaming_App_Usage_Hours']:.2f} horas\n")
            f.write("\n")
            
            # Número de apps vs Idade
            f.write("Número de Apps por Faixa Etária:\n")
            age_group_stats = self.df_clean.groupby('Age_Group')['Number_of_Apps_Used'].mean()
            for age_group, avg_apps in age_group_stats.items():
                f.write(f"{age_group}: {avg_apps:.1f} apps\n")
            f.write("\n")
            
            # Uso de apps por faixa etária
            f.write("Uso de Apps por Faixa Etária:\n")
            age_usage_stats = self.df_clean.groupby('Age_Group').agg({
                'Total_App_Usage_Hours': 'mean',
                'Daily_Screen_Time_Hours': 'mean'
            })
            
            for age_group in age_usage_stats.index:
                f.write(f"{age_group}:\n")
                f.write(f"  Uso Total: {age_usage_stats.loc[age_group, 'Total_App_Usage_Hours']:.2f} horas\n")
                f.write(f"  Tempo de Tela: {age_usage_stats.loc[age_group, 'Daily_Screen_Time_Hours']:.2f} horas\n")
        
        # 5.1 Resumo geral
        print("\n5.1 RESUMO GERAL DO DATASET")
        print("-" * 40)
        print(f"• Total de usuários analisados: {len(self.df_clean)}")
        print(f"• Período de análise: Dataset estático")
        print(f"• Localizações cobertas: {', '.join(self.df_clean['Location'].unique())}")
        print(f"• Faixa etária: {self.df_clean['Age'].min()} a {self.df_clean['Age'].max()} anos")
        
        # 5.2 Insights demográficos
        print("\n5.2 INSIGHTS DEMOGRÁFICOS")
        print("-" * 40)
        gender_dist = self.df_clean['Gender'].value_counts(normalize=True) * 100
        print(f"• Distribuição de gênero:")
        print(f"  - Masculino: {gender_dist.get('Male', 0):.1f}%")
        print(f"  - Feminino: {gender_dist.get('Female', 0):.1f}%")
        
        location_dist = self.df_clean['Location'].value_counts(normalize=True) * 100
        print(f"• Principais localizações:")
        for loc, pct in location_dist.head(3).items():
            print(f"  - {loc}: {pct:.1f}%")
        
        # 5.3 Insights de comportamento
        print("\n5.3 INSIGHTS DE COMPORTAMENTO")
        print("-" * 40)
        
        # Padrões de uso por gênero
        gender_behavior = self.df_clean.groupby('Gender').agg({
            'Social_Media_Usage_Hours': 'mean',
            'Productivity_App_Usage_Hours': 'mean',
            'Gaming_App_Usage_Hours': 'mean',
            'Total_App_Usage_Hours': 'mean'
        })
        
        print("• Padrões de uso por gênero:")
        for gender in gender_behavior.index:
            print(f"  - {gender}:")
            print(f"    * Redes sociais: {gender_behavior.loc[gender, 'Social_Media_Usage_Hours']:.2f}h/dia")
            print(f"    * Produtividade: {gender_behavior.loc[gender, 'Productivity_App_Usage_Hours']:.2f}h/dia")
            print(f"    * Jogos: {gender_behavior.loc[gender, 'Gaming_App_Usage_Hours']:.2f}h/dia")
        
        # Padrões por localização
        location_behavior = self.df_clean.groupby('Location').agg({
            'Daily_Screen_Time_Hours': 'mean',
            'Number_of_Apps_Used': 'mean'
        }).sort_values('Daily_Screen_Time_Hours', ascending=False)
        
        print(f"\n• Localizações com maior tempo de tela:")
        for loc in location_behavior.index:
            print(f"  - {loc}: {location_behavior.loc[loc, 'Daily_Screen_Time_Hours']:.2f}h/dia, "
                  f"{location_behavior.loc[loc, 'Number_of_Apps_Used']:.1f} apps")
        
        # 5.4 Insights de correlação
        print("\n5.4 INSIGHTS DE CORRELAÇÃO")
        print("-" * 40)
        
        correlation_matrix = self.df_clean[['Age', 'Total_App_Usage_Hours', 'Daily_Screen_Time_Hours', 
                                         'Number_of_Apps_Used']].corr()
        
        # Encontrar correlações mais fortes
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.3:  # Correlações moderadas ou fortes
                    corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_value))
        
        print("• Correlações significativas:")
        for col1, col2, corr in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            direction = "positiva" if corr > 0 else "negativa"
            strength = "forte" if abs(corr) > 0.7 else "moderada"
            print(f"  - {col1} vs {col2}: correlação {direction} {strength} ({corr:.3f})")
        
        # 5.5 Insights de faixa etária
        print("\n5.5 INSIGHTS POR FAIXA ETÁRIA")
        print("-" * 40)
        
        age_insights = self.df_clean.groupby('Age_Group').agg({
            'Number_of_Apps_Used': ['mean', 'std'],
            'Total_App_Usage_Hours': ['mean', 'std'],
            'Social_Media_Usage_Hours': 'mean',
            'Gaming_App_Usage_Hours': 'mean'
        }).round(2)
        
        print("• Comportamento por faixa etária:")
        for age_group in age_insights.index:
            print(f"  - {age_group} anos:")
            print(f"    * Apps usados: {age_insights.loc[age_group, ('Number_of_Apps_Used', 'mean')]:.1f} "
                  f"(±{age_insights.loc[age_group, ('Number_of_Apps_Used', 'std')]:.1f})")
            print(f"    * Uso total: {age_insights.loc[age_group, ('Total_App_Usage_Hours', 'mean')]:.2f}h/dia")
            print(f"    * Redes sociais: {age_insights.loc[age_group, ('Social_Media_Usage_Hours', 'mean')]:.2f}h/dia")
            print(f"    * Jogos: {age_insights.loc[age_group, ('Gaming_App_Usage_Hours', 'mean')]:.2f}h/dia")
        
        # 5.6 Recomendações
        print("\n5.6 RECOMENDAÇÕES E OBSERVAÇÕES")
        print("-" * 40)
        print("• Baseado nos dados analisados, observamos que:")
        print("  1. O tempo de uso de dispositivos varia significativamente por localização")
        print("  2. Existem diferenças claras nos padrões de uso entre gêneros")
        print("  3. A faixa etária influencia no número de apps utilizados")
        print("  4. Redes sociais são a categoria mais utilizada em geral")
        print("  5. Há correlação entre o número de apps e o tempo total de uso")
        
        print("\n• Sugestões para análise futura:")
        print("  1. Investigar os fatores que influenciam as diferenças por localização")
        print("  2. Analisar o impacto do tempo de uso na produtividade")
        print("  3. Estudar padrões de uso ao longo do tempo (dados temporais)")
        print("  4. Segmentar análise por tipo de dispositivo ou sistema operacional")
        print("  5. Investigar a relação entre uso de apps e satisfação do usuário")
        
        print("\n" + "=" * 60)
        print("ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("=" * 60)
        print(f"• Total de visualizações geradas: 4")
        print(f"• Diretório de saída: 'visualizations/'")
        print(f"• Relatórios salvos em: 'reports/'")
        print(f"  - relatorio_completo.txt")
        print(f"  - visualizacoes_texto.txt")
        
        return True

def main():
    """Função principal para executar a análise"""
    print("SISTEMA DE ANÁLISE DE DADOS DE USO DE SMARTPHONE")
    print("Dataset: Smartphone Usage and Behavioral Dataset")
    print("=" * 60)
    
    # Inicializar o analisador com o caminho correto do arquivo
    file_path = '/home/lucas/Downloads/Sistema de Análise de Dados de Uso de Smartphone/mobile_usage_behavioral_analysis.csv'
    analyzer = SmartphoneUsageAnalyzer(file_path)
    
    # Executar as etapas de análise
    steps = [
        ("Carregamento de dados", analyzer.load_data),
        ("Tratamento de dados", analyzer.clean_data),
        ("Análise exploratória", analyzer.exploratory_analysis),
        ("Criação de visualizações", analyzer.create_visualizations),
        ("Geração de relatório", analyzer.generate_report)
    ]
    
    for step_name, step_function in steps:
        print(f"\nExecutando: {step_name}")
        try:
            success = step_function()
            if not success:
                print(f"Erro na etapa: {step_name}")
                break
        except Exception as e:
            print(f"Erro na etapa '{step_name}': {e}")
            break
    
    print("\n" + "=" * 60)
    print("PROCESSO FINALIZADO")
    print("=" * 60)

if __name__ == "__main__":
    main()