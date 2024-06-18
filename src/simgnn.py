"""Классы для модулей SimGNN."""

import glob
import torch
import random
import numpy as np
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from layers import AttentionModule, TenorNetworkModule
from utils import process_pair, calculate_loss, calculate_normalized_ged
import networkx as nx
from node2vec import Node2Vec

class SimGNN(torch.nn.Module):
    """
    SimGNN: Нейронный подход к быстрому вычислению схожести графов
    https://arxiv.org/abs/1808.05689
    """
    def __init__(self, args, number_of_labels):
        """
        :param args: Объект аргументов.
        :param number_of_labels: Количество меток узлов.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels + 128
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Определение формы узкого места.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Создание слоев.
        """
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Расчет гистограммы из матрицы сходства.
        :param abstract_features_1: Матрица признаков для графа 1.
        :param abstract_features_2: Матрица признаков для графа 2.
        :return hist: Гистограмма оценок сходства.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        """
        Прохождение свертки.
        :param edge_index: Индексы ребер.
        :param features: Матрица признаков.
        :return features: Абстрактная матрица признаков.
        Здесь создаются начальные эмбеддинги для двух графов
        """

        features = self.convolution_1(features, edge_index)

        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)
        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, data):
        """
        Прямой проход с графами.
        :param data: Словарь данных.
        :return score: Оценка сходства.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        return score

class SimGNNTrainer(object):
    """
    Обучение модели SimGNN.
    """
    def __init__(self, args):
        """
        :param args: Объект аргументов.
        """
        self.args = args
        self.initial_label_enumeration()
        self.train_node2vec()
        self.setup_model()

    def setup_model(self):
        """
        Создание SimGNN.
        """
        self.model = SimGNN(self.args, self.number_of_labels)

    def train_node2vec(self):
        print("\nОбучение модели node2vec.\n")
        all_graphs = self.training_graphs + self.testing_graphs
        all_edges = []
        for graph_file in all_graphs:
            graph_data = process_pair(graph_file)
            all_edges.extend(graph_data["graph_edge_1"] + graph_data["graph_edge_2"])
        all_graph = nx.Graph()
        all_graph.add_edges_from(all_edges)

        self.node2vec = Node2Vec(all_graph, dimensions=128, walk_length=40, num_walks=300, workers=4)
        self.node2vec_model = self.node2vec.fit(window=10, min_count=3, batch_words=8)

    def initial_label_enumeration(self):
        """
        Сбор уникальных идентификаторов узлов.
        """
        print("\nПеречисление уникальных меток.\n")
        self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")
        graph_pairs = self.training_graphs + self.testing_graphs
        self.global_labels = set()
        for graph_pair in tqdm(graph_pairs):
            data = process_pair(graph_pair)
            self.global_labels = self.global_labels.union(set(data["labels_1"]))
            self.global_labels = self.global_labels.union(set(data["labels_2"]))
        self.global_labels = sorted(self.global_labels)
        self.global_labels = {val:index  for index, val in enumerate(self.global_labels)}
        self.number_of_labels = len(self.global_labels)

    def create_batches(self):
        """
        Создание пакетов из списка тренировочных графов.
        :return batches: Список списков с пакетами.
        """
        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph:graph+self.args.batch_size])
        return batches

    def transfer_to_torch(self, data):
        """
        Перевод данных в формат torch и создание хеш-таблицы.
        Включает индексы, признаки и целевую переменную.
        :param data: Словарь данных.
        :return new_data: Словарь тензоров Torch.
        """
        new_data = dict()
        edges_1 = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]
        edges_2 = data["graph_2"] + [[y, x] for x, y in data["graph_2"]]

        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)
        features_1, features_2 = [], []
        for n in data["labels_1"]:
            features_1.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        for n in data["labels_2"]:
            features_2.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])
        #print(data["labels_1"])
        features_1 = torch.FloatTensor(np.array(features_1))
        features_2 = torch.FloatTensor(np.array(features_2))
        #print(features_1)
        #print(data["graph_1_index"])
        #print(data["graph_edge_1"])
        edges_name_1 = data["graph_edge_1"]
        graph_1 = nx.Graph()
        graph_1.add_edges_from(edges_name_1)
        #
        edges_name_2 = data["graph_edge_2"]
        graph_2 = nx.Graph()
        graph_2.add_edges_from(edges_name_2)
        #
        # # Создание графа из всех ребер первого и второго графов
        # all_edges = edges_name_1 + edges_name_2
        # graph_combined = nx.Graph()
        # graph_combined.add_edges_from(all_edges)
        #
        # # Обучение node2vec на объединенном графе
        # node2vec = Node2Vec(graph_combined, dimensions=16, walk_length=30, num_walks=200, workers=4)
        # model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Извлечение эмбеддингов для узлов каждого графа
        embeddings_1 = {node: self.node2vec_model.wv.get_vector(node) for node in graph_1.nodes()}
        embeddings_np_1 = np.array([embeddings_1[node] for node in graph_1.nodes()])
        embedding_tensor_1 = torch.from_numpy(embeddings_np_1)
        new_features_1 = torch.cat((features_1, embedding_tensor_1), dim=1)

        embeddings_2 = {node: self.node2vec_model.wv.get_vector(node) for node in graph_2.nodes()}
        embeddings_np_2 = np.array([embeddings_2[node] for node in graph_2.nodes()])
        embedding_tensor_2 = torch.from_numpy(embeddings_np_2)
        new_features_2 = torch.cat((features_2, embedding_tensor_2), dim=1)

        new_data["edge_index_1"] = edges_1
        new_data["edge_index_2"] = edges_2

        new_data["features_1"] = new_features_1
        new_data["features_2"] = new_features_2

        #new_data["features_1"] = features_1
        #new_data["features_2"] = features_2

        norm_ged = data["ged"] / max(len(data["labels_1"]), len(data["labels_2"]))



        new_data["target"] = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float()

        return new_data

    def process_batch(self, batch):
        """
        Прямой проход с пакетом данных.
        :param batch: Пакет расположений пар графов.
        :return loss: Потеря на пакете.
        """
        self.optimizer.zero_grad()
        losses = 0
        for graph_pair in batch:
            data = process_pair(graph_pair)
            data = self.transfer_to_torch(data)
            prediction = self.model(data)
            prediction = prediction.view(-1)
            losses = losses + torch.nn.functional.mse_loss(data["target"], prediction)
        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def fit(self):
        """
        Обучение модели.
        """
        print("\nОбучение модели.\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc="Эпоха")
        for epoch in epochs:
            batches = self.create_batches()
            self.loss_sum = 0
            main_index = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Пакеты"):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum/main_index
                epochs.set_description("Эпоха (Потеря=%g)" % round(loss, 5))

    def score(self):
        """
        Оценка на тестовом наборе.
        """
        print("\n\nОценка модели.\n")
        self.model.eval()
        self.scores = []
        self.ground_truth = []
        all_predictions = []  # Список для хранения предсказаний всех графов
        all_target = []
        for graph_pair in tqdm(self.testing_graphs):
            data = process_pair(graph_pair)
            self.ground_truth.append(calculate_normalized_ged(data))
            data = self.transfer_to_torch(data)
            target = data["target"]
            prediction = self.model(data)
            all_predictions.append(prediction.item())
            all_target.append(target)
            self.scores.append(calculate_loss(prediction, target))
        self.print_evaluation()
        print("Предсказания модели для всех графов:", all_predictions)
        print("Какие должны быть предсказания:", all_target)

        # Пример кода для вычисления процента сходства между предсказанными значениями и истинными значениями GED
        def calculate_similarity_percentage(predictions):
            similarities = []
            for pred in predictions:
                similarity = pred
                similarities.append(similarity)
            return similarities

        # Пример использования функции для вычисления процента сходства
        similarities = calculate_similarity_percentage(all_predictions)
        for i, similarity in enumerate(similarities):
            print(f"Процент сходства для пары графов {i + 1}: {round(similarity * 100,4)}%")
        def calculate_similarity_percentage_true(predictions):
            similarities = []
            for pred in predictions:
                similarity = pred
                similarities.append(similarity)
            return similarities

        # Пример использования функции для вычисления процента сходства
        similarities_true = calculate_similarity_percentage_true(all_target)
        # Вывод результатов
        for i, similarity in enumerate(similarities_true):
            print(f"Процент сходства для пары графов истина {i + 1}: {similarity * 100}%")

    def print_evaluation(self):
        """
        Вывод оценки ошибки.
        """
        norm_ged_mean = np.mean(self.ground_truth)
        base_error = np.mean([(n-norm_ged_mean)**2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        print("\nБазовая ошибка: " +str(round(base_error, 5))+".")
        print("\nТестовая ошибка модели: " +str(round(model_error, 5))+".")

    def save(self):
        torch.save(self.model.state_dict(), self.args.save_path)

    def load(self):
        self.model.load_state_dict(torch.load(self.args.load_path))
