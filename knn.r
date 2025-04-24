# Carregando a biblioteca tidyverse, que inclui pacotes úteis para manipulação e visualização de dados
library(tidyverse)

# Carregando o dataset CSV com uma especificação de tipos de colunas
dados <- read_csv("4.heart.csv", col_types="nffnnffnfnfnff")

# Visualizando os dados em uma aba separada
View(dados)
# Exibindo a estrutura geral do dataset
glimpse(dados)
# Exibindo estatísticas descritivas para cada coluna
summary(dados)

# Removendo observações que possuem dados faltantes nas colunas especificadas
dados_limpos <- dados %>% 
  filter(!is.na(restingBP) &  # pressão arterial em repouso
           !is.na(cholesterol) &  # nível de colesterol
           !is.na(highBloodSugar) &  # açúcar no sangue
           !is.na(restingHR) &  # frequência cardíaca em repouso
           !is.na(restingECG) &  # resultado do eletrocardiograma em repouso
           !is.na(exerciseAngina) &  # angina induzida por exercício
           !is.na(STdepression) &  # depressão do segmento ST
           !is.na(STslope) &  # inclinação do segmento ST
           !is.na(coloredVessels) &  # número de vasos coloridos por fluoroscopia
           !is.na(defectType)  # tipo de defeito cardíaco
  )
# Exibindo estatísticas descritivas dos dados após remoção dos NA
summary(dados_limpos)

# Função que normaliza os dados entre 0 e 1
normalize <- function(valor) {
  return((valor- min(valor)) / (max(valor) - min(valor)))
}

# Normalizando colunas numéricas selecionadas
dados_normal <-dados_limpos %>% 
  mutate(age = normalize(age)) %>%  # idade
  mutate(restingBP = normalize(restingBP)) %>%  # pressão em repouso
  mutate(cholesterol = normalize(cholesterol)) %>%  # colesterol
  mutate(restingHR = normalize(restingHR)) %>%  # batimentos cardíacos em repouso
  mutate(STdepression = normalize(STdepression)) %>%  # depressão do segmento ST
  mutate(coloredVessels = normalize(coloredVessels))  # vasos coloridos

# Visualizando o resumo dos dados normalizados
summary(dados_normal)

# Convertendo os dados normalizados para um data frame tradicional
dados.df <- data.frame(dados_normal)

# Separando a coluna com o resultado (variável alvo) das demais colunas (variáveis preditoras)
dados.df.resultado <- dados.df %>% select(heartDisease)
dados.df.semresultado <- dados.df %>% select(-heartDisease)

# Visualizando os dois novos data frames
view(dados.df.resultado)
view(dados.df.semresultado)

# Instalando o pacote fastDummies para criação de variáveis dummy (colunas booleanas)
install.packages("fastDummies")
library(fastDummies)

# Criando variáveis dummy para as colunas categóricas
dadosdummies <- dummy_columns(dados.df.semresultado,
                              remove_selected_columns = TRUE )

# Exibindo os nomes das colunas após a transformação
colnames(dadosdummies)
# Exibindo a estrutura dos dados após dummificação
glimpse(dadosdummies)
# Visualizando o novo dataset
view(dadosdummies)

# Definindo uma semente para reprodutibilidade
set.seed(1234)
# Criando um índice aleatório para dividir os dados em treino (75%) e teste (25%)
indice <-sample(nrow(dadosdummies), 
                round(nrow(dadosdummies)*.75), replace = FALSE)
# Subconjunto de treino
dados_treino <- dadosdummies[indice,]
# Subconjunto de teste
dados_teste <- dadosdummies[-indice,]

# Separando os rótulos (resultados) de treino e teste e convertendo para fator
dados_resultado_treino <- as.factor(dados.df.resultado[indice,])
dados_resultado_teste <- as.factor(dados.df.resultado[-indice,])

# Instalando e carregando a biblioteca para KNN
install.packages("class")
library(class)

# Aplicando o algoritmo KNN com k=3
knn_resultado <- knn(train=dados_treino , 
                     test=dados_teste , cl=dados_resultado_treino ,k=3)
# Exibindo os primeiros resultados
head(knn_resultado)
# Criando uma matriz de confusão
knn_matriz <- table(dados_resultado_teste,knn_resultado)
# Exibindo a matriz de confusão
knn_matriz
# Calculando a acurácia do modelo
sum(diag(knn_matriz))/nrow(dados_teste)

# Repetindo o processo com k=15
knn_resultado2 <- knn(train=dados_treino , 
                      test=dados_teste , cl=dados_resultado_treino ,k=15)
head(knn_resultado2)
knn_matriz2 <- table(dados_resultado_teste,knn_resultado2)
knn_matriz2
sum(diag(knn_matriz2))/nrow(dados_teste)

# Repetindo o processo com k=19
knn_resultado3 <- knn(train=dados_treino , 
                      test=dados_teste , cl=dados_resultado_treino ,k=21)
head(knn_resultado3)
knn_matriz3 <- table(dados_resultado_teste,knn_resultado3)
help(knn)
knn_matriz3
sum(diag(knn_matriz3))/nrow(dados_teste)

# Criando um gráfico de dispersão com ggplot
ggplot(dados_limpos, aes(x=age, y=restingHR, color=heartDisease, size=factor(restingHR))) + 
  geom_point(alpha=0.3) +  # adicionando pontos com transparência
  labs(x="Age", y="Resting Bloop Pressure", 
       title="Age vs Bloop pressure for the heart Disease", 
       color= "Heart Disease State") + 
  guides(size="none")  # removendo a legenda para o tamanho dos pontos

# Carregando biblioteca para métricas
install.packages("caret")
library(caret)

# Criando um data frame para armazenar os resultados
resultados_knn <- data.frame(
  k = integer(),
  acuracia = numeric(),
  precisao = numeric(),
  recall = numeric(),
  f1 = numeric()
)

# Testando valores de k de 1 a 30
for (k in 1:30) {
  # Rodando o KNN
  knn_pred <- knn(train = dados_treino,
                  test = dados_teste,
                  cl = dados_resultado_treino,
                  k = k)
  
  # Criando a matriz de confusão
  matriz <- confusionMatrix(knn_pred, dados_resultado_teste)
  
  # Extraindo métricas
  acuracia <- matriz$overall["Accuracy"]
  precisao <- matriz$byClass["Precision"]
  recall <- matriz$byClass["Recall"]
  f1 <- matriz$byClass["F1"]
  
  # Armazenando no data frame
  resultados_knn <- rbind(resultados_knn, 
                          data.frame(k = k,
                                     acuracia = acuracia,
                                     precisao = precisao,
                                     recall = recall,
                                     f1 = f1))
}

# Visualizando os 5 melhores resultados por F1-score
top_f1 <- resultados_knn[order(-resultados_knn$f1), ][1:5, ]
print(top_f1)

# Plotando as métricas por valor de K
library(reshape2)
resultados_long <- melt(resultados_knn, id.vars = "k")

ggplot(resultados_long, aes(x = k, y = value, color = variable)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  labs(title = "Tuning de KNN com Métricas de Desempenho",
       x = "Valor de K",
       y = "Métrica",
       color = "Métrica") +
  theme_minimal()