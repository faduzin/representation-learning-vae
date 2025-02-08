# Exploração do Espaço Latente com Autoencoders Variacionais
Este projeto explora o aprendizado de representações usando Autoencoders Variacionais (VAEs) nos conjuntos de dados Iris e Wine. O foco está em entender como os VAEs comprimem dados tabulares de alta dimensão em representações significativas no espaço latente. Por meio de experimentos com arquiteturas de VAE e Beta-VAE, o projeto visa visualizar e analisar quão bem esse modelo captura a estrutura subjacente dos conjuntos de dados.

## Tabela de Conteúdo

1. [Contexto](#contexto)
2. [O que é um Autoencoder Variacional?](#o-que-é-um-autoencoder-variacional?)
3. [Ferramentas Utilizadas](#ferramentas-utilizadas)
4. [O Processo](#o-processo)
5. [A Análise](#a-análise)
6. [O que Aprendi](#o-que-aprendi)
7. [Habilidades Praticadas](#habilidades-praticadas)
8. [Conclusão](#conclusão)
9. [Contato](#contato)
10. [Contribuições](#contribuições)
11. [Estrutura do Repositório](#estrutura-do-repositório)

## Contexto

Este projeto foi desenvolvido para praticar e aplicar conhecimentos sobre Autoencoders Variacionais (VAEs) no contexto de aprendizado de representações. Ao trabalhar com conjuntos de dados tabulares conhecidos, como Iris e Wine, o objetivo é aprofundar o entendimento de como os VAEs aprendem a comprimir dados em um espaço latente significativo e quão efetivamente essas representações capturam a estrutura subjacente dos dados.

## O que é um Autoencoder Variacional?

Um Autoencoder Variacional (VAE) é um tipo de modelo generativo que aprende a comprimir dados em um espaço latente de menor dimensão e depois reconstruí-los em sua forma original. Diferentemente dos autoencoders tradicionais, que aprendem representações pontuais fixas, os VAEs introduzem uma abordagem probabilística ao aprender uma distribuição sobre o espaço latente. Isso permite que os VAEs não apenas reconstruam dados existentes, mas também gerem novos dados semelhantes ao amostrar dessa distribuição aprendida.

### Como ele funciona?

Um VAE consiste em três componentes principais:

1. **Codificador (Encoder):**  
   O codificador recebe os dados de entrada (por exemplo, recursos dos conjuntos Iris ou Wine) e mapeia para um conjunto de parâmetros — tipicamente uma **média** (\(\mu\)) e **log-variância** (\(\log \sigma^2\)) — que definem uma distribuição Gaussiana no espaço latente. Em vez de codificar os dados em um único ponto, o codificador define uma **distribuição de probabilidade** para cada entrada.

2. **Truque da Reparametrização (Reparameterization Trick):**  
   Para permitir a retropropagação durante o treinamento, os VAEs utilizam uma técnica chamada **truque da reparametrização**. Isso envolve amostrar uma variável aleatória \(\epsilon\) de uma distribuição normal padrão e transformá-la usando a média e a variância aprendidas:  
   \[
   z = \mu + \sigma \times \epsilon
   \]  
   Essa etapa garante que a natureza estocástica da amostragem não interrompa o processo de aprendizado baseado em gradientes.

3. **Decodificador (Decoder):**  
   O decodificador recebe o vetor latente \(z\) amostrado e tenta reconstruir os dados de entrada originais. A qualidade dessa reconstrução ajuda a orientar o processo de aprendizado, garantindo que o espaço latente capture características significativas da entrada.

### O que os VAEs podem fazer?

1. **Aprendizado de Representação:**  
   Os VAEs aprendem representações compactas e significativas dos dados no espaço latente. Essas representações podem revelar padrões e estruturas subjacentes, tornando-as úteis em tarefas como clustering, visualização e detecção de anomalias.

2. **Geração de Dados:**  
   Como os VAEs aprendem uma distribuição sobre o espaço latente, eles podem gerar **novos dados sintéticos** ao amostrar desse espaço. Isso torna os VAEs modelos generativos poderosos, capazes de criar novos exemplos que se assemelham ao conjunto de dados original.

3. **Redução de Dimensionalidade:**  
   Semelhante a técnicas como PCA (Análise de Componentes Principais), os VAEs podem reduzir dados de alta dimensão para dimensões menores, preservando informações importantes. No entanto, os VAEs fornecem uma abordagem mais flexível e não linear para a redução de dimensionalidade.

4. **Detecção de Anomalias:**  
   Ao medir quão bem um VAE consegue reconstruir os dados de entrada, é possível identificar anomalias — pontos de dados que não se ajustam bem à distribuição aprendida normalmente resultam em maiores erros de reconstrução.

Neste projeto, os VAEs serão usados principalmente para **aprendizado de representações**, explorando quão efetivamente conseguem comprimir os conjuntos de dados **Iris** e **Wine** em espaços latentes significativos e quão bem essas representações capturam as estruturas intrínsecas dos dados.

## Ferramentas Utilizadas

- **Linguagem de Programação:** Python 3.12.9  
- **Ambiente Interativo:** Jupyter Notebook  
- **Bibliotecas e Frameworks:**  
  - NumPy, Pandas para manipulação de dados  
  - Matplotlib para visualização  
  - scikit-learn para pré-processamento de dados e métricas  
  - Uma biblioteca de deep learning (por exemplo, TensorFlow/Keras ou PyTorch) para implementação e treinamento dos modelos VAE  
- **Outras Ferramentas:** Git para controle de versão

## O Processo

1. **Carregamento e Pré-Processamento de Dados:**  
   Os conjuntos de dados Iris e Wine são carregados a partir da pasta `data/`. As etapas de pré-processamento incluem normalização e divisão em conjuntos de treinamento e teste.

2. **Arquitetura do Modelo:**  
   - **Encoder (Codificador):** Transforma as variáveis de entrada em parâmetros do espaço latente (média e log-variância).  
   - **Reparametrização:** Implementa o truque para amostrar vetores latentes mantendo a diferenciabilidade.  
   - **Decoder (Decodificador):** Reconstrói a entrada original a partir do vetor latente.

3. **Treinamento:**  
   Os modelos são treinados usando uma função de perda combinada que inclui a perda de reconstrução (para garantir fidelidade aos dados) e a divergência KL (para regularizar o espaço latente). Hiperparâmetros como taxa de aprendizado, tamanho de batch e o peso do termo KL (especialmente no Beta-VAE) são ajustados nos notebooks.

4. **Avaliação:**  
   Os modelos treinados são avaliados por meio de visualizações do espaço latente (por exemplo, gráficos de dispersão coloridos por rótulos de classe) e análise da qualidade de reconstrução. Também são feitas comparações entre as arquiteturas Vanilla VAE e Beta-VAE para avaliar o impacto da força de regularização nas representações aprendidas.

## A Análise

As visualizações armazenadas na pasta `assets/` — como curvas de perda, gráficos do espaço latente e comparações de dados reconstruídos — fornecem evidências empíricas do desempenho dos modelos. Essas imagens dão suporte à avaliação de quão bem os modelos VAE capturam e reconstructem os dados dos conjuntos Iris e Wine.

## Conclusões de Análise Detalhadas

### Análise do Iris

- **Comportamento da Perda:**  
  O gráfico de perda mostra que o modelo aprendeu efetivamente a partir dos dados.
  <img src="assets\iris-assets\iris-loss.png" alt="Iris Loss" width="70%">

- **Visualização do Espaço Latente:**  
  O gráfico do espaço latente não revela clusters bem definidos, mesmo após a estabilização da função de perda.
  <img src="assets\iris-assets\iris-latent-space.png" alt="Latent Space" width="100%">
  
- **Qualidade da Reconstrução:**  
  Os dados reconstruídos aproximam os dados originais. Observa-se que a classe azul tende a se agrupar à esquerda, a classe verde se centraliza e a vermelha se desloca para a direita.
  <img src="assets\iris-assets\iris-reconstructed-pca.png" alt="Reconstructed PCA" width="70%">

### Análise do Wine

- **Padrões de Codificação:**  
  O modelo extraiu com sucesso padrões de codificação, conforme indicado pela estabilização da perda.
  <img src="assets\wine-assets\wine-loss.png" alt="Wine Loss" width="70%">

- **Características de Reconstrução:**  
  A reconstrução produziu dados altamente centralizados e bem agrupados.
  <img src="assets\wine-assets\wine-reconstructed-pca.png" alt="Wine Reconstructed PCA" width="70%">

- **Agrupamento no Espaço Latente:**  
  Não há formação evidente de clusters na visualização do espaço latente.
  <img src="assets\wine-assets\wine-latent-space.png" alt="Wine Latent Space" width="70%">

### Conclusões Gerais de Criação

- **Consistência do Valor de Perda:**  
  O valor da perda deve ser sempre positivo; caso contrário, há um erro na construção do modelo.
  
- **Impacto da Escala dos Dados:**  
  A faixa na qual os dados são escalados afeta diretamente o cálculo da perda. Inicialmente, ao usar padronização (standard scaling) com binary crossentropy, os valores de perda ficavam variando entre positivo e negativo, levando o modelo a aprender uma tendência de centralizar os dados reconstruídos (resultando em valores quase idênticos de saída). A troca para minmax scaling (0 a 1) resolveu esse problema, permitindo que a função de perda orientasse adequadamente o processo de aprendizado e gerasse recursos reconstruídos mais próximos dos dados originais.
  
- **Escolhas de Visualização:**  
  Usar pairplots para conjuntos de dados com muitas variáveis é pouco prático. Portanto, apenas 5 variáveis escolhidas arbitrariamente foram utilizadas nas visualizações, o que se mostrou suficiente para interpretar os resultados.

## O que Aprendi

- **Aprendizado de Representação:**  
  Aprofundei a compreensão de como os VAEs aprendem representações compactas de conjuntos de dados complexos.
  
- **Modelagem Probabilística:**  
  O projeto proporcionou insights sobre os benefícios e desafios de codificar dados como distribuições em vez de pontos fixos.
  
- **Implementação Prática:**  
  Construir, treinar e avaliar modelos de deep learning usando Jupyter notebooks e código modular aprimorou minhas habilidades práticas.
  
- **Visualização e Análise:**  
  A experiência melhorou minha capacidade de visualizar dados de alta dimensão e interpretar efetivamente estruturas de espaço latente.

## Habilidades Praticadas

- Pré-Processamento e Exploração de Dados  
- Design e Implementação de Modelos de Deep Learning  
- Experimentação com Modelos Generativos (Vanilla VAE vs. Beta-VAE)  
- Uso de Python e Jupyter Notebooks para prototipagem e análise  
- Técnicas Eficazes de Visualização de Dados

## Conclusão

Este projeto demonstra o potencial dos Autoencoders Variacionais para um aprendizado de representações eficaz em conjuntos de dados tabulares clássicos. As evidências empíricas (disponíveis na pasta `assets/`) confirmam que, embora os modelos possam aprender representações significativas e reconstruir bem os dados, desafios como o agrupamento no espaço latente e a escala adequada dos dados devem ser abordados. Trabalhos futuros podem incluir a exploração de outros conjuntos de dados, o refinamento das arquiteturas dos modelos ou a aplicação das representações aprendidas em tarefas posteriores, como clustering e detecção de anomalias.

## Contribuições

- Tayenne Euqueres
- William de Oliveira Silva

## Contato

Se tiver alguma pergunta ou sugestão, fique à vontade para entrar em contato:  
[GitHub](https://github.com/faduzin) | [LinkedIn](https://www.linkedin.com/in/ericfadul/) | [eric.fadul@gmail.com](mailto:eric.fadul@gmail.com)

## Estrutura do Repositório

- **assets/**: Contém imagens, figuras e visualizações geradas durante a análise.
- **data/**: Inclui os conjuntos de dados Iris e Wine utilizados no projeto.
- **notebooks/**: Notebooks Jupyter com experimentos, treinamento de modelos e análise.
- **src/**: Código-fonte para definições de modelos, rotinas de treinamento e funções utilitárias.
- **.gitignore**: Especifica arquivos e diretórios que o Git deve ignorar.
- **LICENSE**: Licença MIT sob a qual este projeto é distribuído.
- **README.md**: Este arquivo, fornecendo uma visão geral do projeto e instruções de uso.
