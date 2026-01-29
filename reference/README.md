## Construir um classificador para prever a change de uma pessoa morrer de covid

dataset: link para download - [HERE](https://www.kaggle.com/datasets/meirnizri/covid19-dataset/download?datasetVersionNumber=1)

- About the dataset - [HERE](https://www.kaggle.com/datasets/meirnizri/covid19-dataset)

  - **Explore Dataset**
    - Exploring the dataset with notebook.    ```@andvsilva 2023-02-15```
    - ```Shape dataset Full: observations/rows: 1048575 and columns: 21```
        - **Feature Engineering(FE)** - Creating the target based on condiction, if we have the date, the patient die and otherwise the '9999-99-99' the patient is alive. ```@andvsilva 2023-02-15```
        - **Feature Selection** - Add **SelectKBest(k=20)**
          - Add Random Oversampling Imbalanced Datasets
            &rarr;```Alive    92.66% Died      7.34% Name ALIVE_OR_DIED```
          -  ```RandomOverSampler: Random oversampling involves randomly duplicating examples from the minority class and adding them to the training dataset.``` 
        - **Feature Importance** - **RandomForestClassifier**(DONE)
        - **Modeling** - We start to test the models ```RandomForestClassifier``` and ```CatBoostClassifier```(iterations=100, 1000 are ideal (Next thing to do!)) ```@andvsilva 2023-02-19```
          - **Predictions** from the models, respectively: ```95.18 %``` and ```95.21 %```

### Pipeline:

- **Cleaning**
- **Feature Selection**
- **Modeling**
### To Do - List

- 1 - Fazer estatistica descritiva das variaveis, correlacdo e histograma
  - [x] WIP
  - [ ] DONE
- 2 - Construir as variaveis explicativas
  - [x] WIP
  - [ ] DONE
- 3 - Clusterize os tipos de pacientes e analise os grupos.
  - [x] WIP
  - [ ] DONE
- 4 - Testar os diversos modelos de classificacao estudados em aula.
  - [x] WIP
  - [ ] DONE
- 5 - Apresentar o resultado para cada um dos tipos de modelo. O que garante a qualidade do modelo? Avaliar o Gini, KS e a Curva ROC(treino e validacao)
  - [x] WIP
  - [ ] DONE
- 6 - Para o modelo finalista avaliar a importancia das variaveis. O que determina um risco baixo de doenca cardiaca? E um risco alto?
  - [x] WIP
  - [ ] DONE
- 7 - Refaca os modelos usando PCA
  - [x] WIP
  - [ ] DONE

**What is hierarchical clustering?**

Hierarchical clustering is part of the group of unsupervised learning models known as clustering. This means that we don’t have a defined target variable unlike in traditional regression or classification tasks. The point of this machine learning algorithm, therefore, is to identify distinct clusters of objects that share similar characteristics by using defined distance metrics on the selected variables. Other machine learning algorithms that fit within this family include Kmeans or DBscan.