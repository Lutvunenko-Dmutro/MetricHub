import tkinter as tk                     # import: підключаємо модуль; tkinter: стандартний GUI-фреймворк; as tk: коротке ім'я (аліас) для звертання
from tkinter import ttk                  # from ... import ...: імпортуємо підмодуль ttk (сучасні віджети: вкладки, стилі)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # конкретний клас рендеру matplotlib в Tkinter (вставляє фігуру у віджет)
import matplotlib.pyplot as plt          # pyplot: інтерфейс для побудови графіків; as plt: коротке ім'я
import seaborn as sns                    # seaborn: бібліотека візуалізації; as sns: коротке ім'я
import pandas as pd                      # pandas: табличні дані (DataFrame); as pd: коротке ім'я
import numpy as np                       # numpy: масиви та математика; as np: коротке ім'я
from scipy.cluster.hierarchy import linkage, dendrogram  # з SciPy імпортуємо функції для ієрархічної кластеризації та малювання дендрограм

# ---------- Допоміжні функції ----------
def euclidean_matrix(X):                 # def: оголошення функції; euclidean_matrix: ім'я; (X): параметр — двовимірний масив ознак
    diff = X[:, None, :] - X[None, :, :] # X[...] індексація; : — беремо всі елементи; None додає нову вісь; broadcasting для парних різниць
    return np.sqrt(np.sum(diff**2, axis=2))  # diff**2: квадрат різниць; np.sum(..., axis=2): сума по ознаках; np.sqrt: корінь — евклідова відстань

def weighted_euclidean_matrix(X, w):     # друга функція: зважена евклідова; параметри X (дані) і w (ваги для кожної ознаки)
    # w – масив ваг такої ж довжини, як кількість ознак
    diff = X[:, None, :] - X[None, :, :] # як вище: парні різниці між кожними двома об'єктами
    return np.sqrt(np.sum(w * (diff**2), axis=2))  # w * (...): покоординатне множення на ваги; далі сума та корінь

def manhattan_matrix(X):                 # мангеттенська (city-block) відстань
    diff = np.abs(X[:, None, :] - X[None, :, :])  # np.abs: модуль різниць
    return np.sum(diff, axis=2)          # сума модулів по ознаках — L1-норма

def jaccard_distance_matrix_binary(X):   # відстань Жаккара для бінарних (0/1) векторів
    n = X.shape[0]                        # X.shape: розміри масиву; [0]: кількість об’єктів (рядків)
    D = np.zeros((n, n))                  # створюємо квадратну матрицю n×n, заповнену нулями
    for i in range(n):                    # цикл по рядках (об'єкт i)
        for j in range(n):                # вкладений цикл по стовпцях (об'єкт j)
            a = np.logical_and(X[i] == 1, X[j] == 1).sum()   # a: спільні «1» (перетин) — елементне AND і підрахунок
            b = np.logical_or(X[i] == 1, X[j] == 1).sum()    # b: «1» хоча б в одного (об’єднання) — елементне OR і підрахунок
            D[i, j] = 0 if b == 0 else 1 - a / b             # тернарний оператор: якщо немає жодної «1», відстань 0; інакше 1 - (a/b)
    return D                               # повертаємо матрицю відстаней

def add_task_tab(notebook, title, data_df, dist_mat, show_scatter=True, show_dendrogram=True):
    # add_task_tab: утиліта для створення вкладки з графіками; notebook: ttk.Notebook; title: заголовок вкладки;
    # data_df: DataFrame з ознаками (x1,x2 чи бінарні); dist_mat: матриця відстаней для heatmap;
    # show_scatter/show_dendrogram: прапорці, чи малювати відповідні панелі
    frame = tk.Frame(notebook)            # tk.Frame: контейнер для вкладки
    notebook.add(frame, text=title)       # додаємо нову вкладку з назвою

    # Готуємо фігуру з підграфіками: до 3 панелей
    n_cols = 3 if (show_scatter and show_dendrogram) else (2 if (show_scatter or show_dendrogram) else 1)
    # умовний вираз: якщо і scatter, і dendrogram — 3 панелі; якщо один з них — 2; інакше — 1
    fig, axes = plt.subplots(1, n_cols, figsize=(16, 5))  # створюємо фігуру з 1 рядком і n_cols стовпцями
    if n_cols == 1:
        axes = [axes]                     # нормалізуємо до списку для єдиної логіки індексації

    col_i = 0                             # індекс поточної панелі (стовпця)

    # Scatter
    if show_scatter:                      # якщо треба малювати scatter
        ax = axes[col_i]                  # беремо поточну вісь
        ax.scatter(data_df.iloc[:, 0], data_df.iloc[:, 1], s=80, c='royalblue')  # точки: перші 2 колонки (x1, x2); s: розмір; c: колір
        for idx, (x, y) in data_df.iterrows():             # ітеруємо рядки DataFrame: idx — індекс (object), (x,y) — значення
            ax.text(x + 0.3, y + 0.3, str(idx), fontsize=10) # підписуємо кожну точку її номером (з невеликим зсувом)
        ax.set_title("Scatter (x1, x2)") # заголовок панелі
        ax.set_xlabel("x1")              # підпис осі X
        ax.set_ylabel("x2")              # підпис осі Y
        ax.grid(True, linestyle='--', alpha=0.4)           # вмикаємо сітку
        col_i += 1                        # переходимо до наступної панелі

    # Heatmap
    ax = axes[col_i]                      # вісь для теплокарти
    dist_df = pd.DataFrame(np.round(dist_mat, 2), index=data_df.index, columns=data_df.index)
    # створюємо DataFrame з матриці відстаней; np.round(..., 2): округлення до 2 знаків; індекси/колонки — номери об’єктів
    sns.heatmap(dist_df, annot=True, fmt=".2f", cmap="mako", linewidths=0.5,
                cbar_kws={'label': 'Distance'}, ax=ax)     # heatmap: annot=True показує числа; fmt: формат; cmap: палітра; cbar_kws: підпис шкали
    ax.set_title("Матриця відстаней")     # заголовок панелі
    ax.set_xlabel("Об’єкт j")             # підпис осі X (колонки матриці)
    ax.set_ylabel("Об’єкт i")             # підпис осі Y (рядки матриці)
    col_i += 1                            # переходимо до наступної панелі

    # Dendrogram (по ознаках x1,x2 — Ward linkage)
    if show_dendrogram:                   # якщо треба малювати дендрограму
        ax = axes[col_i]                  # вісь для дендрограми
        X = data_df.to_numpy()            # перетворюємо DataFrame у масив NumPy (ознаки по стовпцях)
        Z = linkage(X, method='ward')     # linkage: обчислення ієрархічних зв'язків; method='ward': метод Уорда (мінімізація дисперсії)
        dendrogram(Z, labels=data_df.index.to_list(), leaf_font_size=10, ax=ax)  # малюємо дендрограму з підписами об’єктів
        ax.set_title("Дендрограма (Ward linkage)")          # заголовок
        ax.set_xlabel("Об’єкти")           # підпис осі X (мітки листків)
        ax.set_ylabel("Відстань")          # підпис осі Y (висота злиття кластерів)

    fig.tight_layout()                    # автоматично підганяє розмітку, щоб елементи не накладались

    canvas = FigureCanvasTkAgg(fig, master=frame)           # створюємо «полотно» для вставки matplotlib-фігури у Tkinter
    canvas.get_tk_widget().pack(fill='both', expand=True)   # додаємо віджет у фрейм; fill='both': заповнює, expand=True: розтягується

# ---------- Дані для завдань ----------
# Завдання 1 — Табл. 2.10
data1 = pd.DataFrame({                   # pd.DataFrame: створюємо таблицю (DataFrame) з словника
    'object': [1,2,3,4,5,6,7,8],         # ключ 'object': список індексів об'єктів
    'x1': [119.4, 121.0, 16.6, 114.2, 115.8, 15.2, 17.9, 117.5],  # ключ 'x1': значення першої ознаки
    'x2': [16.6, 18.1, 15.5, 19.4, 23.2, 16.7, 15.7, 15.2]        # ключ 'x2': значення другої ознаки
}).set_index('object')                    # set_index('object'): робимо колонку 'object' індексом рядків

# Завдання 2 — Табл. 2.11
data2 = pd.DataFrame({
    'object': [1,2,3,4,5,6,7,8],
    'x1': [73.2, 60.2, 63.7, 70.6, 95.1, 75.8, 93.4, 50.5],
    'x2': [12.2, 11.6, 1.6, 13.7, 16.1, 11.1, 16.5, 1.2]
}).set_index('object')

# Завдання 3 — Табл. 2.12 (ваги w1=0.3, w2=0.7)
data3 = pd.DataFrame({
    'object': [1,2,3,4,5,6,7,8],
    'x1': [114.4, 116.0, 11.6, 19.2, 110.8, 11.2, 12.9, 112.5],
    'x2': [12.6, 14.1, 12.5, 15.4, 19.2, 11.7, 12.7, 12.2]
}).set_index('object')
weights = np.array([0.3, 0.7])           # np.array: вектор ваг; 0.3 для x1, 0.7 для x2

# Завдання 4 — Табл. 2.13 (city-block/мангеттен)
data4 = pd.DataFrame({
    'object': [1,2,3,4,5,6,7,8],
    'x1': [133.2, 120.2, 133.7, 120.6, 115.1, 145.8, 153.4, 137.5],
    'x2': [24.2, 20.6, 16.6, 36.7, 35.1, 72.1, 56.5, 54.2]
}).set_index('object')

# Завдання 5 — Табл. 2.14 (Жаккар, бінарні 0/1)
data5 = pd.DataFrame({
    'object': [1,2,3,4,5],
    'x1': [1,1,0,0,1],
    'x2': [1,1,0,1,1],
    'x3': [0,1,1,0,1]
}).set_index('object')

# ---------- Обчислення матриць ----------
X1 = data1[['x1','x2']].to_numpy()       # [['x1','x2']]: вибираємо дві колонки; to_numpy(): в масив NumPy
M1 = euclidean_matrix(X1)                 # виклик нашої функції: матриця евклідових відстаней для завдання 1

X2 = data2[['x1','x2']].to_numpy()
M2 = euclidean_matrix(X2)                 # матриця евклідових відстаней для завдання 2

X3 = data3[['x1','x2']].to_numpy()
M3 = weighted_euclidean_matrix(X3, weights)  # зважена евклідова: передаємо X3 та вектор weights

X4 = data4[['x1','x2']].to_numpy()
M4 = manhattan_matrix(X4)                 # мангеттенська (city-block) матриця відстаней

X5 = data5.to_numpy()                     # для Жаккара беремо всі колонки (бінарні ознаки)
M5 = jaccard_distance_matrix_binary(X5)   # матриця відстаней Жаккара

# ---------- Tkinter GUI ----------
root = tk.Tk()                            # tk.Tk(): створюємо головне вікно програми
root.title("Практичні завдання 2 — матриці відстаней")  # title(): заголовок вікна

notebook = ttk.Notebook(root)             # ttk.Notebook: виджет «вкладки»
notebook.pack(fill='both', expand=True)   # pack: менеджер розміщення; fill='both': заповнює; expand=True: розтягується

# Додаємо вкладки
add_task_tab(                              # виклик нашої утиліти додавання вкладки
    notebook,                              # notebook: куди додавати
    "Завдання 1 (Евклідова)",              # title: текст на вкладці
    data1[['x1','x2']],                    # data_df: дані для scatter/dendrogram (дві ознаки)
    M1,                                    # dist_mat: матриця для heatmap
    show_scatter=True,                     # показувати scatter
    show_dendrogram=True                   # показувати дендрограму
)

add_task_tab(
    notebook,
    "Завдання 2 (Евклідова)",
    data2[['x1','x2']],
    M2,
    show_scatter=True,
    show_dendrogram=True
)

add_task_tab(
    notebook,
    "Завдання 3 (Зважена Евклідова)",
    data3[['x1','x2']],
    M3,
    show_scatter=True,
    show_dendrogram=True
)

add_task_tab(
    notebook,
    "Завдання 4 (Manhattan)",
    data4[['x1','x2']],
    M4,
    show_scatter=True,
    show_dendrogram=True
)

add_task_tab(
    notebook,
    "Завдання 5 (Жаккар)",
    data5,                                 # тут вся таблиця (три бінарні ознаки)
    M5,
    show_scatter=False,                    # scatter вимкнено (бінарні дані)
    show_dendrogram=False                  # дендрограма вимкнена (не доречно для Жаккара тут)
)

# Додаємо пояснювальну вкладку
exp_frame = tk.Frame(notebook)             # новий контейнер для вкладки «Explanation»
notebook.add(exp_frame, text="Explanation")# додаємо як вкладку
figE, axE = plt.subplots(figsize=(9, 6))   # створюємо фігуру matplotlib і вісь
axE.axis('off')                            # axis('off'): ховаємо осі — будемо писати текст

axE.text(0.02, 0.92, "Пояснення методів", fontsize=16, weight='bold')  # text(x,y, ...): малюємо текст у координатах фігури

axE.text(0.02, 0.82, r"Евклідова:  $d(i,j) = \sqrt{\sum_{k} (x_{ki}-x_{kj})^{2}}$", fontsize=14)
# r"...": сирий рядок (щоб LaTeX не ламався); $...$: LaTeX-формула

axE.text(0.02, 0.75, r"Зважена евклідова:  $d(i,j) = \sqrt{\sum_{k} w_{k} (x_{ki}-x_{kj})^{2}}$", fontsize=14)
axE.text(0.02, 0.68, r"Manhattan (city-block):  $d(i,j) = \sum_{k} |x_{ki}-x_{kj}|$", fontsize=14)
axE.text(0.02, 0.61, r"Відстань Жаккара (бінарні):  $d(i,j) = 1 - \frac{|X_i \cap X_j|}{|X_i \cup X_j|}$", fontsize=14)

axE.text(0.02, 0.48, "Поради:", fontsize=14, weight='bold')
axE.text(0.03, 0.42, "- Якщо масштаби ознак різні, варто нормалізувати/стандартизувати.", fontsize=12)
axE.text(0.03, 0.37, "- Дендрограма будується за ознаками (Ward). Теплокарти показують матриці відстаней.", fontsize=12)

figE.tight_layout()                        # підганяємо розмітку, щоб все красиво розмістилось
canvasE = FigureCanvasTkAgg(figE, master=exp_frame)       # створюємо «полотно» для фігури у Tkinter
canvasE.get_tk_widget().pack(fill='both', expand=True)    # додаємо віджет у вкладку й розтягуємо

root.mainloop()                            # запускаємо головний цикл Tkinter (вікно працює, доки його не закриють)