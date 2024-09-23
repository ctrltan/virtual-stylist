import pandas as pd
import numpy as np
import ast
import random
import math
from feedback import FeedbackMachine
from itertools import combinations
from feature_extractor import FeatureExtractor



class Outfit:
    
    def __init__(self, outfit_row, cluster_index):
        self.reward = 8
        self.cluster_index = cluster_index
        self.outfit_row = outfit_row

    def __str__(self):
        return f"Reward: {self.reward} \nPath: {self.outfit_row['Path']} \nColour 1: {self.outfit_row['Prominent_Colour1']} \nColour 2: {self.outfit_row['Prominent_Colour2']} \nColour 3: {self.outfit_row['Prominent_Colour3']}"
    
    def set_reward_share(self, num):
        self.reward = num

    def get_reward(self):
        return self.reward
    
    def get_outfit(self):
        return self.outfit_row
    
    def get_index(self):
        return self.outfit_row['Unnamed: 0']
    
    def get_cluster_index(self):
        return self.cluster_index


class Inspiration:

    def __init__(self, timestep, outfits=[]):
        self.indexed_outfits = {}
        self.outfits = outfits
        for outfit_tuple in outfits:
            outfit = outfit_tuple[0]
            cluster_index = outfit_tuple[1]
            self.indexed_outfits[outfit['Unnamed: 0']] = Outfit(outfit, cluster_index)
        
        self.outfit_rows = [outfit.get_outfit() for outfit in list(self.indexed_outfits.values())]
        self.outfit_indexes = tuple(sorted(self.indexed_outfits.keys()))
        self.overall_score = 0
        self.timestep = timestep

    def __hash__(self):
        return hash(self.outfit_indexes)
    
    def __eq__(self, other):
        return hasattr(other, 'outfits_indexes') and self.outfit_indexes == other.outfit_indexes
    
    def __str__(self):
        outfit_strings = ""
        for outfit in list(self.indexed_outfits.values()):
            outfit_strings += f"\n\n{outfit}"
        return f"Inspiration consisting of {len(self.outfits)} outfits: {outfit_strings}"
    
    def get_timestep(self):
        return self.timestep
    
    def get_outfits(self):
        return list(self.indexed_outfits.values())
    
    def get_outfit_rows(self):
        return self.outfit_rows
    
    def get_outfit_indexes(self):
        return self.outfit_indexes
    
    def get_outfit_row(self, outfit_index):
        return self.indexed_outfits[outfit_index].get_outfit()
    
    def get_outfit_cluster(self, outfit_index):
        return self.indexed_outfits[outfit_index].get_cluster_index()
    
    def get_overall_score(self):
        return self.overall_score
    
    def set_overall_score(self, score):
        self.overall_score = score
    
    def get_outfit_reward(self, outfit_index):
        return self.indexed_outfits[outfit_index].get_reward()
    
    def size(self):
        return len(self.outfits)
    
    def calculate_rewards(self, item_scores):
        total_rewards = []
        for (key, value) in item_scores.items():
            total_rewards.extend(value)

        total_rewards = sum(total_rewards)

        if total_rewards == 0:
            for key in item_scores.keys():
                self.indexed_outfits[key].set_reward_share(0)
        else:
            for (key, value) in item_scores.items():
                rewards = []
                for rank in value:
                    reward = (rank / total_rewards) * self.overall_score
                    rewards.append(reward)
                mean_reward = np.mean(rewards)
                self.indexed_outfits[key].set_reward_share(mean_reward)

    def contains_outfit(self, outfit_index):
        return outfit_index in self.outfit_indexes
    
    def new_repeat(self, new_timestep):
        return Inspiration(timestep=new_timestep, outfits=self.outfits)
    
    def add_new(self, new_outfit, new_timestep=None):
        if new_timestep == None:
            new_timestep = self.timestep
        new_inspiration = self.outfits + [new_outfit]
        return Inspiration(new_timestep, new_inspiration)
    
    def remove(self, outfit_index, new_timestep=None):
        if new_timestep == None:
            new_timestep = self.timestep
        new_inspiration = [outfit for outfit in self.outfits if outfit[0]['Unnamed: 0'] != outfit_index]
        return Inspiration(new_timestep, new_inspiration)

    




class Assembler:

    VALID_COMBINATIONS = [['tops', 'skirts'], ['tops', 'skirts', 'jackets'], ['tops', 'long_skirts'], ['tops', 'long_skirts', 'jackets'], ['tops', 'trousers'], 
                    ['tops', 'trousers', 'jackets'], ['dresses'], ['dresses', 'jackets']]


    def __init__(self):
        self.outfit_clusters = None
        self.cluster_centroids = {}
        self.outfit_clustering()
        self.all_inspirations = []
        self.max_inspiration_size = 5
        self.current_combo = None
        self.timestep = 0
        self.discount_factor = 0.3
        self.max_epsilon = 0.7
        self.epsilon = 0.7
        self.decay_rate = 0.97
        self.min_epsilon = 0.1
        self.clothing_item_dict = pd.read_csv('clothing_items_csv.csv', 
                                              converters={'Prominent_Colour1': self.tuple_evaluator, 
                                                          'Prominent_Colour2': self.tuple_evaluator, 
                                                          'Prominent_Colour3': self.tuple_evaluator}).to_dict(orient='records')
        self.top_inspiration = None
        self.greatest_inspirations = set()
        self.make_match()
        
    def add_inspiration(self, inspiration: Inspiration):
        self.all_inspirations.append(inspiration)

        if self.top_inspiration == None:
            self.top_inspiration = inspiration
        
        if inspiration.get_overall_score() > 7:
            self.greatest_inspirations.add(inspiration)
            if inspiration.get_overall_score() >= self.top_inspiration.get_overall_score():
                self.top_inspiration = inspiration

        

    def find_euclidean_distance(self, feature_row, centroid):
        squares = []
        for i in range(2, len(centroid)):
            squares.append((feature_row[i] - centroid[i])**2)

        distance = math.sqrt(sum(squares))
        return distance


    def get_feature_matrix(self, dict_list):
        feature_mat = []
        for i in range(len(dict_list)):
            features = [dict_list[i]['Unnamed: 0']]
            values_list = list(dict_list[i].values())[2:]

            for value in values_list:
                try:
                    features.extend(list(value))
                except:
                    features.extend(value.toList())
            
            feature_mat.append(features)

        return feature_mat


    def numpy_evaluator(self, string):
        return np.fromstring(string[1:-1], sep=' ', dtype=float)


    def tuple_evaluator(self, string):
        try:
            return ast.literal_eval(string)
        except:
            return string


    def outfit_clustering(self):
        outfits_dataframe = pd.read_csv('outfits_csv.csv', converters={'Prominent_Colour1': self.tuple_evaluator, 'Prominent_Colour2': self.tuple_evaluator, 'Prominent_Colour3': self.tuple_evaluator,
                                                            'Silhouette':self.numpy_evaluator, 'Presentation_Date': self.tuple_evaluator})
        outfits = outfits_dataframe.to_dict(orient='records')

        feature_mat = self.get_feature_matrix(outfits)

        k = 5

        centroids = random.sample(feature_mat, k)

        clusters = [[] for x in range(k)]

        max_iterations = 30
        iterations = 0

        while iterations <= max_iterations:
            clusters = [[] for x in range(k)]

            iterations += 1
            
            for data_point in feature_mat:
                distances = [self.find_euclidean_distance(data_point[1:], centroid[1:]) for centroid in centroids]
                shortest_distance_index = np.argmin(distances)

                clusters[shortest_distance_index].append(data_point)

            centroids = []

            for feature_cluster in clusters:
                centroid = []
                for i in range(len(feature_cluster[0])):
                    sum = 0
                    for feature_row in feature_cluster:
                        sum += feature_row[i]
                    preserved_type = type(sum)
                    mean = preserved_type(sum / len(feature_cluster))
                    centroid.append(mean)
                centroids.append(centroid)

            


        self.outfit_clusters = []

        for num in range(len(clusters)):
            centroid = centroids[num]
            self.cluster_centroids[num] = centroid
            cluster = clusters[num]
            outfit_cluster = []
            distances = {}
            for data_point in cluster:
                distances[self.find_euclidean_distance(data_point[1:], centroid[1:])] = outfits[data_point[0]]
            sorted_distances = sorted(distances.items())
            outfit_cluster = [value for (key, value) in sorted_distances]
            self.outfit_clusters.append(outfit_cluster)
        

    def make_match(self, next_inspiration=None):

        print(f'\nExploration Rate: {self.epsilon}')
        if next_inspiration == None:
            rand_cluster_index = random.randint(0, len(self.outfit_clusters)-1)
            self.current_combo = Inspiration(self.timestep, [(self.outfit_clusters[rand_cluster_index][0], rand_cluster_index)])
        else:
            self.current_combo = next_inspiration

        produced_items = {}

        for outfit in self.current_combo.get_outfit_rows():
            outfit_colours = [outfit['Prominent_Colour1'], outfit['Prominent_Colour2'], outfit['Prominent_Colour3']]
            colour_filtered_items = self.filter_by_colours(self.clothing_item_dict, outfit_colours)

            try:
                produced_items[outfit['Unnamed: 0']] = self.filter_by_item(colour_filtered_items, outfit['Path'])
            except:
                produced_items[outfit['Unnamed: 0']] = colour_filtered_items



        result = self.assembly(produced_items)
        final_outfit = result[0]
        final_inspirations = result[1]
        feedback = FeedbackMachine(final_outfit, final_inspirations)


        new_inspiration = Inspiration(timestep=self.timestep)
        for outfit_index in final_inspirations:
            if self.current_combo.contains_outfit(outfit_index) and new_inspiration.contains_outfit(outfit_index) == False:
                new_inspiration = new_inspiration.add_new(new_outfit=(self.current_combo.get_outfit_row(outfit_index), self.current_combo.get_outfit_cluster(outfit_index)))

        
        new_inspiration.set_overall_score(feedback.get_overall_score())
        new_inspiration.calculate_rewards(feedback.get_feedback())

        self.add_inspiration(new_inspiration)
        
        self.exploration_strategy(new_inspiration)

    
    def softmax_probabilities(self, values):
        e_values = np.exp(values)
        probabilities = e_values / np.sum(e_values)
        return probabilities
    
    
    def decrease_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.decay_rate

    def increase_epsilon(self):
        if self.epsilon < self.max_epsilon:
            self.epsilon = self.epsilon / self.decay_rate
    
    def exploration_strategy(self, inspiration: Inspiration):
        current_score = inspiration.get_overall_score()
        num_outfits = inspiration.size()
        rewards = np.array([self.get_cumulative_reward(index) for index in inspiration.get_outfit_indexes()])
        lowest_reward_index = np.argmin(rewards)
        lowest_outfit_index = inspiration.get_outfit_indexes()[lowest_reward_index]
        self.timestep += 1

        if random.random() < self.epsilon:
            self.decrease_epsilon()
            print("\nExplore")

            if current_score < 6:
                new_inspiration = Inspiration(timestep=self.timestep)
                num_outfits = random.randint(1, self.max_inspiration_size)

                for i in range(num_outfits):
                    random_cluster_index = random.randint(0, len(self.outfit_clusters)-1)
                    outfits = self.outfit_clusters[random_cluster_index]
                    new_outfit = random.choice(outfits)

                    new_inspiration = new_inspiration.add_new(new_outfit=(new_outfit, random_cluster_index))
                
                return self.make_match(new_inspiration)

            else:
                random_outfit_index = random.choice(list(inspiration.get_outfit_indexes()))

                cluster_index = inspiration.get_outfit_cluster(random_outfit_index)
                new_outfit = random.choice(self.outfit_clusters[cluster_index])

                if inspiration.size() < self.max_inspiration_size:
                    new_inspiration = inspiration.add_new(new_outfit=(new_outfit, cluster_index), new_timestep=self.timestep)
                else:
                    new_inspiration = inspiration.remove(random_outfit_index, self.timestep)
                    new_inspiration = new_inspiration.add_new(new_outfit=(new_outfit, cluster_index))

                return self.make_match(new_inspiration)
        else:
            print("\nExploit")
            """
            exploitation
            """
            
            if current_score < 4:
                self.increase_epsilon()
                temp_inspiration = inspiration.remove(lowest_outfit_index, self.timestep)
                highest_reward_index = np.argmax([self.get_cumulative_reward(outfit_index) for outfit_index in self.top_inspiration.get_outfit_indexes()])
                outfit_index = self.top_inspiration.get_outfit_indexes()[highest_reward_index]
                highest_rewarding_outfit = self.top_inspiration.get_outfit_row(outfit_index)

                new_inspiration = temp_inspiration.add_new(new_outfit=(highest_rewarding_outfit, self.top_inspiration.get_outfit_cluster(outfit_index)))
                return self.make_match(new_inspiration)
            
            self.decrease_epsilon()
            if current_score < 7:
                outfit = self.get_a_top_outfit()
                if inspiration.size() < self.max_inspiration_size:
                    new_inspiration = inspiration.add_new(outfit, self.timestep)
                    return self.make_match(new_inspiration)
                else:
                    temp_inspiration = inspiration.remove(lowest_outfit_index, self.timestep)
                    new_inspiration = temp_inspiration.add_new(outfit, self.timestep)
                    return self.make_match(new_inspiration)
            else:
                random_top_inspiration = random.choice(list(self.greatest_inspirations))
                new_inspiration = random_top_inspiration.new_repeat(self.timestep)
                return self.make_match(new_inspiration)
                
    
    def get_cumulative_reward(self, outfit_index):
        rewards = np.array([inspiration.get_outfit_reward(outfit_index) for inspiration in self.all_inspirations if inspiration.contains_outfit(outfit_index)])
        if len(rewards) == 0: 
            return self.current_combo.get_outfit_reward(outfit_index)
        
        timesteps = np.array([inspiration.get_timestep() for inspiration in self.all_inspirations if inspiration.contains_outfit(outfit_index)])
        exponents = timesteps[::-1]
        discounts = self.discount_factor ** exponents
        discounted_rewards = rewards * discounts

        cumulative_reward = np.sum(discounted_rewards)

        return cumulative_reward
