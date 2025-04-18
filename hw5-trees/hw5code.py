import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ

    thresholds = np.sort(list(feature_vector))
    thresholds = (thresholds[:-1] + thresholds[1 :]) / 2

    H_R_left = np.cumsum(target_vector[:-1]) # а вообще говоря они зеркальны а потому можно посчитать один массив и его развернуть
    H_R_right = np.sum(target_vector) - np.cumsum(target_vector[:-1])
    H_R_left = H_R_left / np.arange(1, len(H_R_left)+1)
    H_R_right = H_R_right / np.arange(len(H_R_right), 0, -1)
    H_R_left = np.array(list(map(lambda x : 1 - x**2 - (1-x)**2, H_R_left)))
    H_R_right = np.array(list(map(lambda x : 1 - x**2 - (1-x)**2, H_R_right)))
    ginis = - H_R_left * (np.arange(1, len(H_R_left)+1) / len(target_vector)) - H_R_right * (np.arange(len(H_R_right), 0, -1) / len(target_vector))
    gini_best = np.max(ginis)
    threshold_best = thresholds[np.argmin(ginis)]
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types # типы фичей
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y != sub_y[0]): # если все объекты в листе одного класса, то останавливаемся
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]): # (1, )? для каждой фичи
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real": # здесь просто значения
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical": # здесь будем возвращать номера соответствующих классов
                counts = Counter(sub_X[:, feature]) # по сути крутой словарь
                clicks = Counter(sub_X[sub_y == 1, feature]) # те эл-ты что sub_y == 1
                ratio = {}
                for key, current_count in counts.items(): # список пар значение - количество
                    if key in clicks:
                        current_click = clicks[key] # получим кол-во эл-ов данного класса, что sub_y == 1
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count # доля положительных внутри класса
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))) # таким сложным образом просто получаем упорядоченные ключи категорий по значениям в ratio
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories))))) # перенумерованный словарь

                feature_vector = np.array(map(lambda x: categories_map[x], sub_X[:, feature])) # вектор фичей, т.е. номера соответствующих значению признака категорий
            else:
                raise ValueError

            if len(feature_vector) == 1: #? 3
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best: # если нашли более хороший критерий ветвления
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold # булевый массив чтобы получить эл-ты которые идут в левое и правое поддеревья

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "Categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items()))) # все элементы вектора фичей ушедшие налево
                else:
                    raise ValueError

        if feature_best is None: #?
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        best_feature = node["feature_split"]
        if self._feature_types[best_feature] == "real":
            if x[best_feature] < node["threshold"]:
                self._predict_node(x, node["left_child"])
            else:
                self._predict_node(x, node["right_child"])
        elif self._feature_types[best_feature] == "categorical":
            if x[best_feature] in node["categories_split"]:
                self._predict_node(x, node["left_child"])
            else:
                self._predict_node(x, node["right_child"])
        else:
            raise ValueError
            

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
