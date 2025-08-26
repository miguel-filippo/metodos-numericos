## Cálculo Numérico — Projetos 1, 2 e 3

Este repositório reúne três projetos práticos desenvolvidos na disciplina de Cálculo Numérico, cobrindo interpolação polinomial, regressão linear com seleção de variáveis e uma visualização interativa de Gradiente Descendente (GD) versus Gradiente Descendente Estocástico (SGD).

—

### Projeto 1 — Interpolação e curvatura de uma viga (Lagrange/Hermite)

- Questão‑problema: Dado um conjunto reduzido de medições de deflexão de uma viga em três pontos e informações de derivada em outros dois pontos, reconstruir a curva de deflexão e localizar a região de maior curvatura.
- Como foi resolvido: Montou‑se um polinômio interpolador de grau 4 impondo condições mistas de valor (y) e derivada (y'), caracterizando uma interpolação do tipo Hermite. O sistema linear dos coeficientes é resolvido com álgebra linear numérica. Em seguida:
	- calcula‑se a curvatura via segunda derivada do polinômio;
	- maximiza‑se numericamente a curvatura no intervalo usando busca delimitada;
	- compara‑se a curva obtida com um polinômio “original” de referência e gera‑se gráfico de deflexão e de curvatura.
- Pontos fortes da solução:
	- Usa informações adicionais de derivada para maior fidelidade local sem extrapolar grau excessivo;
	- Implementação estável e direta (sistema 5×5);
	- Visualizações claras para validar a reconstrução e o ponto de máxima curvatura.
- Como executar rapidamente:
	1) Instalar dependências: `pip install -r "Projeto 1/requirements.txt"`
	2) Rodar: `python "Projeto 1/lagrange_interpolation.py"`
	3) Saídas: `Projeto 1/lagrange_interpolation.png` (deflexão e curvatura).

—

### Projeto 2 — Regressão linear com engenharia de atributos e seleção de variáveis (Concreto)

- Questão‑problema: Prever a resistência do concreto a partir de suas composições (treino e teste em `dados/Concreto - treino.csv` e `dados/Concreto - teste.csv`). Alcançar bom erro fora da amostra com um modelo simples e interpretável.
- Como foi resolvido: Construiu‑se um pipeline de Mínimos Quadrados com:
	- engenharia de atributos (para cada feature: x, x², x³, log|x| e todas as interações par‑a‑par);
	- normalização Z‑score;
	- seleção exaustiva do melhor subconjunto com R ∈ {1,2,3,4} via k‑fold CV (k=5);
	- ajuste dos coeficientes por BFGS (quase‑Newton) e comparação com GD batch;
	- geração de predições e gráficos de diagnóstico (dispersões, diferenças de θ e predições).
	O código principal está em `Projeto 2/model_tools.py` (funções utilitárias, treino, CV e geração de predições). Há scripts auxiliares:
	- `correlational.py`: matriz de correlação das features engenheiradas com o alvo;
	- `best_previsions.py`: instancia o melhor modelo (R=4 por padrão) e salva predições em `previsoes/best_model_predictions.csv`;
	- `calculate_mse.py`: calcula MSE entre predições fornecidas.
- Pontos fortes da solução:
	- Bom compromisso viés‑variância ao limitar R e usar validação cruzada;
	- Modelo linear interpretável, com engenharia de atributos para capturar não linearidades;
	- Dois otimizadores (BFGS e GD) para robustez e verificação de consistência;
	- Reprodutibilidade e organização do pipeline (salva figuras e saídas padronizadas).
- Como executar rapidamente:
	1) Instalar dependências: `pip install -r "Projeto 2/requirements.txt"`
	2) Rodar o pipeline completo: `python "Projeto 2/model_tools.py"`
	3) Gerar predições do melhor modelo: `python "Projeto 2/best_previsions.py"`
	4) Saídas: arquivos em `Projeto 2/previsoes/` e gráficos em `Projeto 2/graficos/`.

—

### Projeto 3 — Visualização interativa: GD vs SGD em regressão linear (Pygame)

- Questão‑problema: Construir uma ferramenta didática para entender, de forma visual, como o Gradiente Descendente (GD) e o Gradiente Descendente Estocástico (SGD) ajustam uma reta de regressão a partir de pontos definidos pelo usuário.
- Como foi resolvido: Desenvolveu‑se uma aplicação interativa em Pygame. O usuário clica para adicionar pontos; ao iniciar, a reta é ajustada em tempo real por GD ou SGD, com normalização interna dos dados para estabilidade numérica e cálculo do MSE a cada iteração. Botões permitem limpar pontos, alternar método e começar/parar o ajuste.
- Pontos fortes da solução:
	- Aprendizado por experimentação visual e imediata;
	- Comparação lado a lado entre GD e SGD, evidenciando suas dinâmicas;
	- Normalização automática e desenho robusto da reta no sistema cartesiano.
- Como executar rapidamente:
	1) Instalar dependências: `pip install -r "Projeto 3/requirements.txt"`
	2) Rodar a aplicação: `python "Projeto 3/main.py"`
	3) Uso: clique para adicionar pontos; “Começar” para ajustar; “GD/SGD” alterna método; “Limpar pontos” reinicia.

—

Desenvolvido por mim: Miguel Filippo Rocha Calhabeu — USP (ICMC) — para a matéria de Cálculo Numérico.
